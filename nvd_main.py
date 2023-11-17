import clip
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
import kornia
import torch
from nvd_neural_style_field import NvdNeuralStyleField
from utils import device 
from nvd_render import NvdRenderer
from nvd_mesh import NvdMesh
from Normalization import NvdMeshNormalizer
import numpy as np
import random
import copy
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
from nvdiffmodeling.src import obj as nvdObj
from nvdiffmodeling.src import mesh as nvdMesh
from nvdiffmodeling.src import texture as nvdTexture
import xatlas

def run_branched(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Constrain all sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Load CLIP model 
    clip_model, preprocess = clip.load(args.clipmodel, device, jit=args.jit)
    
    # Adjust output resolution depending on model type 
    res = 224 
    if args.clipmodel == "ViT-L/14@336px":
        res = 336
    if args.clipmodel == "RN50x4":
        res = 288
    if args.clipmodel == "RN50x16":
        res = 384
    if args.clipmodel == "RN50x64":
        res = 448
        
    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
    # Check that isn't already done
    if (not args.overwrite) and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        print(f"Already done with {args.output_dir}")
        exit()
    elif args.overwrite and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        import shutil
        for filename in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    dir = args.output_dir

    render = NvdRenderer(dim=(res, res), rast_backend=args.rast_backend)

    tmp_mesh = nvdObj.load_obj(args.obj_path)
    tmp_mesh = nvdMesh.unit_size(tmp_mesh)
    #original_vertices = tmp_mesh.v_pos.clone()
    #original_normals = tmp_mesh.v_nrm.clone()

    numpy_vertices = tmp_mesh.v_pos.detach().cpu().numpy()
    numpy_faces = tmp_mesh.t_pos_idx.detach().cpu().numpy()
    numpy_normals = tmp_mesh.v_nrm.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(numpy_vertices, numpy_faces, numpy_normals)
    # export to obj
    temp_obj_path = os.path.join(dir, f"original_xatlas.obj")
    xatlas.export(temp_obj_path, numpy_vertices[vmapping], indices, uvs, numpy_normals[vmapping])

    # call load_obj again
    nvd_mesh = nvdObj.load_obj(temp_obj_path)
    nvd_mesh = nvdMesh.unit_size(nvd_mesh)

    nvd_prior_color = torch.full(size=(nvd_mesh.v_pos.shape[0], 3), fill_value=0.5, device=device)
    texture_res = 128
    texture_coords = torch.mul(nvd_mesh.v_tex, (texture_res - 1))
    texture_coords = texture_coords.long().to(device)
    nvd_normal_map = nvdTexture.create_trainable(np.array([0, 0, 1]), [texture_res]*2, True)
    nvd_specular_map = nvdTexture.create_trainable(np.array([0, 0, 0]), [texture_res]*2, True)

    background = None
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background)
        background = torch.tile(background, (1, res, res, 1)).to(device)

    losses = []

    n_augs = args.n_augs
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # CLIP Transform
    clip_transform = transforms.Compose([
        transforms.Resize((res, res)),
        clip_normalizer
    ])

    # Augmentation settings
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])

    # Augmentations for normal network
    if args.cropforward :
        curcrop = args.normmincrop
    else:
        curcrop = args.normmaxcrop
    normaugment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])
    cropiter = 0
    cropupdate = 0
    if args.normmincrop < args.normmaxcrop and args.cropsteps > 0:
        cropiter = round(args.n_iter / (args.cropsteps + 1))
        cropupdate = (args.maxcrop - args.mincrop) / cropiter

        if not args.cropforward:
            cropupdate *= -1

    # Displacement-only augmentations
    displaugment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(args.normmincrop, args.normmincrop)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])

    normweight = 1.0

    # MLP Settings
    input_dim = 6 if args.input_normals else 3
    if args.only_z:
        input_dim = 1
    mlp = NvdNeuralStyleField(args.sigma, args.depth, args.width, 'gaussian', args.colordepth, args.normdepth,
                                args.normratio, args.clamp, args.normclamp, niter=args.n_iter,
                                progressive_encoding=args.pe, input_dim=input_dim, exclude=args.exclude).to(device)
    mlp.reset_weights()

    optim = torch.optim.Adam(mlp.parameters(), args.learning_rate, weight_decay=args.decay)
    activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.lr_decay)
    if not args.no_prompt:
        if args.prompt:
            prompt = ' '.join(args.prompt)
            prompt_token = clip.tokenize([prompt]).to(device)
            encoded_text = clip_model.encode_text(prompt_token)

            # Save prompt
            with open(os.path.join(dir, prompt), "w") as f:
                f.write("")

            # Same with normprompt
            norm_encoded = encoded_text
    if args.normprompt is not None:
        prompt = ' '.join(args.normprompt)
        prompt_token = clip.tokenize([prompt]).to(device)
        norm_encoded = clip_model.encode_text(prompt_token)

        # Save prompt
        with open(os.path.join(dir, f"NORM {prompt}"), "w") as f:
            f.write("")

    if args.image:
        img = Image.open(args.image)
        img = preprocess(img).to(device)
        encoded_image = clip_model.encode_image(img.unsqueeze(0))
        if args.no_prompt:
            norm_encoded = encoded_image

    loss_check = None

    nvd_vertices = copy.deepcopy(nvd_mesh.v_pos)
    nvd_network_input = copy.deepcopy(nvd_vertices)

    #vmapping = torch.from_numpy(vmapping.astype(np.int64)).long()
    #nvd_normals = copy.deepcopy(original_normals)
    #nvd_vertices = copy.deepcopy(original_vertices)
    #nvd_network_input = copy.deepcopy(nvd_vertices)

    if args.symmetry == True:
        nvd_network_input[:,2] = torch.abs(nvd_network_input[:,2])

    if args.standardize == True:
        # Each channel into z-score
        nvd_network_input = (nvd_network_input - torch.mean(nvd_network_input, dim=0))/torch.std(nvd_network_input, dim=0)

    for i in tqdm(range(args.n_iter)):
        optim.zero_grad()

        nvd_sampled_mesh = nvd_mesh
        nvd_sampled_mesh, nvd_pred_rgb, nvd_pred_normal = nvd_update_mesh(mlp, nvd_network_input, nvd_prior_color, nvd_sampled_mesh, nvd_vertices, nvd_normal_map, nvd_specular_map, texture_coords, texture_res)
        #nvd_sampled_mesh, nvd_pred_rgb, nvd_pred_normal = nvd_update_mesh_vmapping(mlp, nvd_network_input, nvd_prior_color, nvd_sampled_mesh, nvd_vertices, nvd_normals, nvd_normal_map, nvd_specular_map, texture_coords, texture_res, vmapping)

        rendered_images, elev, azim = render.nvd_render_front_views(nvd_sampled_mesh, num_views=args.n_views,
                                                                show=args.show,
                                                                center_azim=args.frontview_center[0],
                                                                center_elev=args.frontview_center[1],
                                                                std=args.frontview_std,
                                                                return_views=True,
                                                                background=background)
    
        if n_augs == 0:
            clip_image = clip_transform(rendered_images)
            encoded_renders = clip_model.encode_image(clip_image)
            if not args.no_prompt:
                loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

        # Check augmentation steps
        if args.cropsteps != 0 and cropupdate != 0 and i != 0 and i % args.cropsteps == 0:
            curcrop += cropupdate
            # print(curcrop)
            normaugment_transform = transforms.Compose([
                transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
                transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                clip_normalizer
            ])

        if n_augs > 0:
            loss = 0.0
            for _ in range(n_augs):
                augmented_image = augment_transform(rendered_images)
                encoded_renders = clip_model.encode_image(augmented_image)
                if not args.no_prompt:
                    if args.prompt:
                        if args.clipavg == "view":
                            if encoded_text.shape[0] > 1:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                torch.mean(encoded_text, dim=0), dim=0)
                            else:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                encoded_text)
                        else:
                            loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                if args.image:
                    if encoded_image.shape[0] > 1:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                        torch.mean(encoded_image, dim=0), dim=0)
                    else:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                        encoded_image)
        if args.splitnormloss:
            for param in mlp.mlp_normal.parameters():
                param.requires_grad = False
        loss.backward(retain_graph=True)

        # Normal augment transform
        if args.n_normaugs > 0:
            normloss = 0.0
            for _ in range(args.n_normaugs):
                augmented_image = normaugment_transform(rendered_images)
                encoded_renders = clip_model.encode_image(augmented_image)
                if not args.no_prompt:
                    if args.prompt:
                        if args.clipavg == "view":
                            if norm_encoded.shape[0] > 1:
                                normloss -= normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                                 torch.mean(norm_encoded, dim=0),
                                                                                 dim=0)
                            else:
                                normloss -= normweight * torch.cosine_similarity(
                                    torch.mean(encoded_renders, dim=0, keepdim=True),
                                    norm_encoded)
                        else:
                            normloss -= normweight * torch.mean(
                                torch.cosine_similarity(encoded_renders, norm_encoded))
                if args.image:
                    if encoded_image.shape[0] > 1:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                        torch.mean(encoded_image, dim=0), dim=0)
                    else:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                        encoded_image)
            if args.splitnormloss:
                for param in mlp.mlp_normal.parameters():
                    param.requires_grad = True
            if args.splitcolorloss:
                for param in mlp.mlp_rgb.parameters():
                    param.requires_grad = False
            if not args.no_prompt:
                normloss.backward(retain_graph=True)

        # Also run separate loss on the uncolored displacements
        if args.geoloss:
            texture = torch.full(size=(texture_res, texture_res, 3), fill_value=0.5, dtype=torch.float32, device=device)
            texture_map = nvdTexture.Texture2D(texture)
            geoloss_mesh = nvdMesh.Mesh(
                material={
                    'bsdf': 'diffuse',
                    'kd': texture_map,
                    'ks': nvd_specular_map,
                    'normal': nvd_normal_map,
                },
                base=nvd_sampled_mesh
            )
            geo_renders, elev, azim = render.nvd_render_front_views(geoloss_mesh, num_views=args.n_views,
                                                                show=args.show,
                                                                center_azim=args.frontview_center[0],
                                                                center_elev=args.frontview_center[1],
                                                                std=args.frontview_std,
                                                                return_views=True,
                                                                background=background)
            if args.n_normaugs > 0:
                normloss = 0.0
                ### avgview != aug
                for _ in range(args.n_normaugs):
                    augmented_image = displaugment_transform(geo_renders)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if norm_encoded.shape[0] > 1:
                        normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(norm_encoded, dim=0), dim=0)
                    else:
                        normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            norm_encoded)
                    if args.image:
                        if encoded_image.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)
                normloss.backward(retain_graph=True)

        optim.step()

        for param in mlp.mlp_normal.parameters():
            param.requires_grad = True
        for param in mlp.mlp_rgb.parameters():
            param.requires_grad = True

        if activate_scheduler:
            lr_scheduler.step()

        with torch.no_grad():
            losses.append(loss.item())

        # Adjust normweight if set
        if args.decayfreq is not None:
            if i % args.decayfreq == 0:
                normweight *= args.cropdecay

        if i % 100 == 0:
            report_process(args, dir, i, loss, loss_check, losses, rendered_images)

    #final_mesh, pred_rgb, pred_normal = nvd_update_mesh_vmapping(mlp, nvd_network_input, nvd_prior_color, nvd_mesh, nvd_vertices, nvd_normals, nvd_normal_map, nvd_specular_map, texture_coords, texture_res, vmapping)
    final_mesh, pred_rgb, pred_normal = nvd_update_mesh(mlp, nvd_network_input, nvd_prior_color, nvd_mesh, nvd_vertices, nvd_normal_map, nvd_specular_map, texture_coords, texture_res)
    nvd_export_final_results(args, dir, losses, final_mesh, pred_rgb, pred_normal, 720, render)



def report_process(args, dir, i, loss, loss_check, losses, rendered_images):
    print('iter: {} loss: {}'.format(i, loss.item()))
    torchvision.utils.save_image(rendered_images, os.path.join(dir, 'iter_{}.jpg'.format(i)))
    if args.lr_plateau and loss_check is not None:
        new_loss_check = np.mean(losses[-100:])
        # If avg loss increased or plateaued then reduce LR
        if new_loss_check >= loss_check:
            for g in torch.optim.param_groups:
                g['lr'] *= 0.5
        loss_check = new_loss_check

    elif args.lr_plateau and loss_check is None and len(losses) >= 100:
        loss_check = np.mean(losses[-100:])


def nvd_export_final_results(args, dir, losses, final_mesh, pred_rgb, pred_normal, resolution, renderer):
    with torch.no_grad():
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        nvdObj.write_obj(
            str(dir),
            final_mesh.eval()
        )

        # Run renders
        if args.save_render:
            background = None
            if args.background is not None:
                assert len(args.background) == 3
                background = torch.tensor(args.background)
                background = torch.tile(background, (1, resolution, resolution, 1)).to(device)
                
            save_rendered_results(dir, final_mesh, args.frontview_center[1], args.frontview_center[0], background, resolution, renderer)
            
        # Save final losses
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))

def save_rendered_results(dir, mesh, center_elev, center_azim, background, resolution, renderer):
    img = renderer.nvd_render_single_view(mesh, center_elev, center_azim, background, resolution)
    img = img[0].cpu()
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"final_cluster.png"))


def update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices):
    pred_rgb, pred_normal = mlp(network_input)
    # NCHORTEK TODO: Find a replacement for index_vertices_by_faces (if its needed?)
    # really what I need to do is figure out how to handle vertex colors
    sampled_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    sampled_mesh.vertices = vertices + sampled_mesh.vertex_normals * pred_normal
    NvdMeshNormalizer(sampled_mesh)()


def nvd_update_mesh(mlp, nvd_network_input, nvd_prior_color, nvd_sampled_mesh, nvd_vertices, normal_map, specular_map, texture_coords, texture_res):
    # Get predicted color and vertex shifts from our mlp
    nvd_pred_rgb, nvd_pred_normal = mlp(nvd_network_input)
    assert not torch.isnan(nvd_pred_rgb).any()
    assert not torch.isnan(nvd_pred_normal).any()
    # Calculate new vertex positons, scaled along the normal direction by a value predicted by our mlp
    vertex_positions = nvd_vertices + nvd_sampled_mesh.v_nrm * nvd_pred_normal

    # calculate new per-vertex colors by shifting from [0.5, 0.5, 0.5] by values predicted by our mlp
    vertex_colors = nvd_prior_color + nvd_pred_rgb

    texture = torch.full(size=(texture_res, texture_res, 3), fill_value=0.5, dtype=torch.float32, device=device)
    texture[texture_coords[:,1], texture_coords[:,0]] = vertex_colors

    """
    print("----------------------------------------")
    print("nvd_network_input.shape")
    print(nvd_network_input.shape)
    print("nvd_pred_rgb.shape")
    print(nvd_pred_rgb.shape)
    print("nvd_pred_normal.shape")
    print(nvd_pred_normal.shape)
    print("nvd_sampled_mesh.v_nrm.shape")
    print(nvd_sampled_mesh.v_nrm.shape)
    print("nvd_vertices.shape")
    print(nvd_vertices.shape)
    print("nvd_prior_color.shape")
    print(nvd_prior_color.shape)
    print("vertex_positions.shape")
    print(vertex_positions.shape)
    print("vertex_colors.shape")
    print(vertex_colors.shape)
    print("texture_coords.shape")
    print(texture_coords.shape)
    print("texture.shape init")
    print(texture.shape)
    print("texture.shape end")
    print(texture.shape)
    print("----------------------------------------")
    """

    texture_map = nvdTexture.Texture2D(texture)

    """
    ready_texture = nvdTexture.Texture2D(
        kornia.filters.gaussian_blur2d(
            texture_map.data.permute(0, 3, 1, 2),
            kernel_size=(7, 7),
            sigma=(3, 3),
        ).permute(0, 2, 3, 1).contiguous()
    )

    ready_specular = nvdTexture.Texture2D(
        kornia.filters.gaussian_blur2d(
            specular_map.data.permute(0, 3, 1, 2),
            kernel_size=(7, 7),
            sigma=(3, 3),
        ).permute(0, 2, 3, 1).contiguous()
    )

    ready_normal = nvdTexture.Texture2D(
        kornia.filters.gaussian_blur2d(
            normal_map.data.permute(0, 3, 1, 2),
            kernel_size=(7, 7),
            sigma=(3, 3),
        ).permute(0, 2, 3, 1).contiguous()
    )
    """
    
    nvd_sampled_mesh = nvdMesh.Mesh(
        v_pos=vertex_positions,
        material={
            'bsdf': 'diffuse',
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=nvd_sampled_mesh
    )

    nvd_sampled_mesh = nvdMesh.auto_normals(nvd_sampled_mesh)
    nvd_sampled_mesh = nvdMesh.compute_tangents(nvd_sampled_mesh).eval()
    nvd_sampled_mesh = nvdMesh.unit_size(nvd_sampled_mesh)

    return nvd_sampled_mesh, nvd_pred_rgb, nvd_pred_normal

def nvd_update_mesh_vmapping(mlp, nvd_network_input, nvd_prior_color, nvd_sampled_mesh, nvd_vertices, nvd_normals, normal_map, specular_map, texture_coords, texture_res, vmapping):
    # Get predicted color and vertex shifts from our mlp
    nvd_pred_rgb, nvd_pred_normal = mlp(nvd_network_input)
    assert not torch.isnan(nvd_pred_rgb).any()
    assert not torch.isnan(nvd_pred_normal).any()
    # Calculate new vertex positons, scaled along the normal direction by a value predicted by our mlp
    vertex_positions = nvd_vertices[vmapping] + nvd_normals[vmapping] * nvd_pred_normal[vmapping]

    # calculate new per-vertex colors by shifting from [0.5, 0.5, 0.5] by values predicted by our mlp
    vertex_colors = nvd_prior_color + nvd_pred_rgb[vmapping]

    texture = torch.full(size=(texture_res, texture_res, 3), fill_value=0.5, dtype=torch.float32, device=device)
    texture[texture_coords[:,1], texture_coords[:,0]] = vertex_colors

    texture_map = nvdTexture.Texture2D(texture)
    
    nvd_sampled_mesh = nvdMesh.Mesh(
        v_pos=vertex_positions,
        material={
            'bsdf': 'diffuse',
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=nvd_sampled_mesh
    )

    #nvd_sampled_mesh = nvdMesh.auto_normals(nvd_sampled_mesh)
    #nvd_sampled_mesh = nvdMesh.compute_tangents(nvd_sampled_mesh).eval()
    #nvd_sampled_mesh = nvdMesh.unit_size(nvd_sampled_mesh)

    return nvd_sampled_mesh, nvd_pred_rgb, nvd_pred_normal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='meshes/mesh1.obj')
    parser.add_argument('--prompt', nargs="+", default='a pig with pants')
    parser.add_argument('--normprompt', nargs="+", default=None)
    parser.add_argument('--promptlist', nargs="+", default=None)
    parser.add_argument('--normpromptlist', nargs="+", default=None)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='round2/alpha5')
    parser.add_argument('--traintype', type=str, default="shared")
    parser.add_argument('--sigma', type=float, default=10.0)
    parser.add_argument('--normsigma', type=float, default=10.0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--colordepth', type=int, default=2)
    parser.add_argument('--normdepth', type=int, default=2)
    parser.add_argument('--normwidth', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--normal_learning_rate', type=float, default=0.0005)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--no_pe', dest='pe', default=True, action='store_false')
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--n_views', type=int, default=5)
    parser.add_argument('--n_augs', type=int, default=0)
    parser.add_argument('--n_normaugs', type=int, default=0)
    parser.add_argument('--n_iter', type=int, default=6000)
    parser.add_argument('--encoding', type=str, default='gaussian')
    parser.add_argument('--normencoding', type=str, default='xyz')
    parser.add_argument('--layernorm', action="store_true")
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--clamp', type=str, default="tanh")
    parser.add_argument('--normclamp', type=str, default="tanh")
    parser.add_argument('--normratio', type=float, default=0.05)
    parser.add_argument('--frontview', action='store_true')
    parser.add_argument('--no_prompt', default=False, action='store_true')
    parser.add_argument('--exclude', type=int, default=0)

    # Training settings 
    parser.add_argument('--frontview_std', type=float, default=8)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--clipavg', type=str, default=None)
    parser.add_argument('--geoloss', action="store_true")
    parser.add_argument('--samplebary', action="store_true")
    parser.add_argument('--promptviews', nargs="+", default=None)
    parser.add_argument('--mincrop', type=float, default=1)
    parser.add_argument('--maxcrop', type=float, default=1)
    parser.add_argument('--normmincrop', type=float, default=0.1)
    parser.add_argument('--normmaxcrop', type=float, default=0.1)
    parser.add_argument('--splitnormloss', action="store_true")
    parser.add_argument('--splitcolorloss', action="store_true")
    parser.add_argument("--nonorm", action="store_true")
    parser.add_argument('--cropsteps', type=int, default=0)
    parser.add_argument('--cropforward', action='store_true')
    parser.add_argument('--cropdecay', type=float, default=1.0)
    parser.add_argument('--decayfreq', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--background', nargs=3, type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_render', action="store_true")
    parser.add_argument('--input_normals', default=False, action='store_true')
    parser.add_argument('--symmetry', default=False, action='store_true')
    parser.add_argument('--only_z', default=False, action='store_true')
    parser.add_argument('--standardize', default=False, action='store_true')

    # CLIP model settings 
    parser.add_argument('--clipmodel', type=str, default='ViT-B/32')
    parser.add_argument('--jit', action="store_true")

    # Differential Renderer Settings
    parser.add_argument('--rast_backend', type=str, default='cuda')
    
    args = parser.parse_args()

    run_branched(args)
