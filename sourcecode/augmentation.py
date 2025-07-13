import os
import cv2
import albumentations as A

def augment_images_albumentations(input_folders, output_folder, augment_num_per_image_dict):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.7),
        A.RandomScale(scale_limit=0.2, p=0.5),
    ])
    
    for input_folder in input_folders:
        class_name = os.path.basename(input_folder)  
        output_class_folder = os.path.join(output_folder, class_name)
        os.makedirs(output_class_folder, exist_ok=True)

        augment_num_per_image = augment_num_per_image_dict.get(class_name, 0)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Failed to read {img_path}")
                    continue
                if img.shape[2] != 4:
                    print(f"Warning: Not RGBA image: {img_path}")
                    continue
                base_name = os.path.splitext(filename)[0]

                for i in range(augment_num_per_image):
                    augmented = transform(image=img)
                    aug_img = augmented["image"]
                    save_path = os.path.join(output_class_folder, f"{base_name}_aug{i+1}.png")
                    cv2.imwrite(save_path, aug_img)
