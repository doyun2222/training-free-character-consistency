from dreamsim import dreamsim
from PIL import Image
import os
from pathlib import Path
from itertools import combinations
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse
import clip

import sys
sys.path.append("/home/cig_project/acc/code/image-background-remove-tool")

from carvekit.api.high import HiInterface
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import sklearn.preprocessing

interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)


def replace_bg_with_noise(image_path):
    np.random.seed(42)

    image = interface([image_path])[0].convert('RGB')
    image_array = np.array(image)

    # Identify regions with all black pixels
    black_pixels = np.all(image_array == [130, 130, 130], axis=-1)

    # Generate random uniform noise
    noise = np.random.randint(0, 256, size=image_array.shape, dtype=np.uint8)

    # Replace black regions with noise
    image_array[black_pixels] = noise[black_pixels]

    # Convert the modified NumPy array back to a PIL image
    image = Image.fromarray(image_array)

    return image


def get_clip_score_for_image_text(image, caption, model, device):
    # Define the preprocessing steps for the image
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Load and preprocess the image
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Tokenize and encode the caption
    caption = 'A photo depicts ' + caption if not caption.startswith('A photo depicts') else caption
    caption_tokens = clip.tokenize(caption, truncate=True).squeeze()
    caption_tokens = caption_tokens.unsqueeze(0)  # Add batch dimension

    # Move image and caption tokens to the device
    image = image.to(device)
    caption_tokens = caption_tokens.to(device)

    # Encode image and caption
    image_features = model.encode_image(image)
    text_features = model.encode_text(caption_tokens)

    # Normalize features
    image_features = sklearn.preprocessing.normalize(image_features.cpu().detach().numpy(), axis=1)
    text_features = sklearn.preprocessing.normalize(text_features.cpu().detach().numpy(), axis=1)

    # Calculate CLIP score
    clip_score = 2.5 * np.sum(image_features * text_features)

    return clip_score


def calculate_avg_clip_text_score(folder_path, single_folder):
    model, _ = clip.load('ViT-B/32', device="cuda", jit=False)
    model.eval()
    if single_folder:
        subfolders = [folder_path]
    else:
        subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    avg_distances = {}
    for subfolder_name in subfolders:
        images = []
        text_prompts = []
        avg_distance = 0
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for f in Path(subfolder_path).rglob('*'):
            if f.suffix.lower() in {'.png', '.jpg'}:
                images.append(Image.open(f))
                text_prompts.append(os.path.basename(f))

        # Skip if no images are found
        if not images:
            print(f"No images found in {subfolder_name}, skipping...")
            continue

        for image, text_prompt in zip(images, text_prompts):
            distance = get_clip_score_for_image_text(image, text_prompt, model, "cuda")
            avg_distance += distance.item()

        avg_distance /= len(images)
        avg_distances[subfolder_name] = avg_distance
    return avg_distances


# Compute DreamSim distance between two images
def compute_dreamsim_distance(image1, image2, model, processor):
    features1 = processor(image1).to("cuda")
    features2 = processor(image2).to("cuda")
    distance = model(features1, features2)
    return distance


# Compute CLIP distance between two images
def compute_clip_distance(image1, image2, model, processor):
    inputs1 = processor(images=image1, return_tensors="pt")
    with torch.no_grad():
        features1 = model.get_image_features(**inputs1)

    inputs2 = processor(images=image2, return_tensors="pt")
    with torch.no_grad():
        features2 = model.get_image_features(**inputs2)

    distance = torch.nn.functional.cosine_similarity(features1, features2)
    return distance


# Compute average distance for images in a folder
def calculate_avg_distance(folder_path, mode, remove_bg, single_folder):
    if mode == "dreamsim":
        model, processor = dreamsim(pretrained=True)
    elif mode == "clip_image":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    if single_folder:
        subfolders = [folder_path]
    else:
        subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    avg_distances = {}
    for subfolder_name in subfolders:
        images = []
        avg_distance = 0
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for f in Path(subfolder_path).rglob('*'):
            if f.suffix.lower() in {'.png', '.jpg'}:
                if remove_bg:
                    images.append(replace_bg_with_noise(f))
                else:
                    images.append(Image.open(f))

        # Skip if no images are found
        if not images:
            print(f"No images found in {subfolder_name}, skipping...")
            continue

        if len(images) < 2:
            avg_distances[subfolder_name] = float('nan')
            continue

        for combo in combinations(images, 2):
            if mode == "dreamsim":
                distance = compute_dreamsim_distance(combo[0], combo[1], model, processor)
            elif mode == "clip_image":
                distance = compute_clip_distance(combo[0], combo[1], model, processor)

            avg_distance += distance.item()

        avg_distance /= len(images) * (len(images) - 1) / 2
        avg_distances[subfolder_name] = avg_distance
        print(f"Average distance for {subfolder_name}: {avg_distance}")

    return avg_distances


# Main function to parse arguments and execute tasks
def main():
    parser = argparse.ArgumentParser(description="Calculate image similarities using DreamSim or CLIP.")
    parser.add_argument('--mode', type=str, choices=['dreamsim', 'clip_image', 'clip_text'],
                        help='Mode of operation: "dreamsim" or "clip_image"')
    parser.add_argument('--folder', type=str, help='Path to the folder containing image subfolders')
    parser.add_argument('--image1', type=str, help='Path to the first image for pairwise comparison')
    parser.add_argument('--image2', type=str, help='Path to the second image for pairwise comparison')
    parser.add_argument('--image_clip_text', type=str,
                        help='Path to the image for clip_text score, image_name as text_prompt')
    parser.add_argument('--remove_background', action='store_true', help='Remove background from images')
    parser.add_argument('--single_folder', action='store_true', help='single_folder')

    args = parser.parse_args()

    if args.image_clip_text:
        image = Image.open(args.image_clip_text).convert("RGB")
        caption = os.path.basename(args.image_clip_text)
        model, _ = clip.load('ViT-B/32', device="cuda", jit=False)
        model.eval()
        clip_score = get_clip_score_for_image_text(image, caption, model, "cuda")

        print(f"Clip_score:{clip_score}")

    elif args.image1 and args.image2:
        if args.remove_background == True:
            image1 = replace_bg_with_noise(args.image1)
            image2 = replace_bg_with_noise(args.image2)
        else:
            image1 = Image.open(args.image1).convert("RGB")
            image2 = Image.open(args.image2).convert("RGB")

        if args.mode == 'dreamsim':
            model, processor = dreamsim(pretrained=True)
            distance = compute_dreamsim_distance(image1, image2, model, processor)
        elif args.mode == 'clip_image':
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            distance = compute_clip_distance(image1, image2, model, processor)

        print(f"Distance: {distance}")

    else:
        if args.mode == 'dreamsim' or args.mode == 'clip_image':
            avg_distances = calculate_avg_distance(args.folder, args.mode, args.remove_background, args.single_folder)
        elif args.mode == 'clip_text':
            avg_distances = calculate_avg_clip_text_score(args.folder, args.single_folder)

        total_avg_distance = 0
        for subfolder, avg_distance in avg_distances.items():
            total_avg_distance += round(avg_distance, 5)
            #if avg_distance < 0.84:
            print(f"Average distance for {subfolder}: {avg_distance}")

        total_avg_distance /= len(avg_distances)
        print(f"{args.mode}_total_avg_distance: {total_avg_distance}")


if __name__ == "__main__":
    main()
