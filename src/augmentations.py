import albumentations as alb
import numpy as np

# Crop augmentations


def _convert_to_random_scale_tuple(input):

    if isinstance(input, (float, int)):
        input = (input, input)

    if not isinstance(input, np.ndarray):
        input = np.array(input)

    return input


def create_scaling_crop(scale_factor, image_resize=2048, patch_size=512, crop="random"):

    assert crop in ["random", "center", None]

    if crop == "random":
        crop = alb.RandomCrop(patch_size, patch_size, p=1.0)
    elif crop == "center":
        crop = alb.CenterCrop(patch_size, patch_size, p=1.0)
    else:
        crop = alb.Identity()

    scale_factor = _convert_to_random_scale_tuple(scale_factor)

    # alb.RandomScale uses a bias of 1. For whatever reason. So scaling the image by 0.25 actually requires the input -0.75 wtf. Oh and it wants tuples.
    scale_factor -= 1
    scale_factor = tuple(scale_factor)

    scale_only = alb.Compose([alb.augmentations.geometric.resize.SmallestMaxSize(max_size=image_resize, interpolation=2, always_apply=False, p=1.0),
                              alb.RandomScale(scale_limit=scale_factor, p=1.0),
                              crop], p=1.0)

    return scale_only


def create_zoom_crop(zoom_factor, image_resize=2048, patch_size=512, crop="random"):

    zoom_factor = _convert_to_random_scale_tuple(zoom_factor)

    scale_factor = (zoom_factor * patch_size)/image_resize

    aug = create_scaling_crop(scale_factor, image_resize, patch_size, crop)
    return aug


def create_fraction_of_border_crop(border_fraction, image_resize=2048, patch_size=512, crop="random"):

    border_fraction = _convert_to_random_scale_tuple(border_fraction)
    zoom_factor = 1/border_fraction

    return create_zoom_crop(zoom_factor=zoom_factor, image_resize=image_resize, patch_size=patch_size, crop=crop)


def create_fraction_of_image_crop(image_fraction, image_resize=2048, patch_size=512, crop="random"):

    image_fraction = _convert_to_random_scale_tuple(image_fraction)

    border_length_fraction = np.sqrt(image_fraction)

    return create_fraction_of_border_crop(border_fraction=border_length_fraction, image_resize=image_resize, patch_size=patch_size, crop=crop)


# Augmentation Strategies
tellez_transforms_train = alb.Compose([

    create_fraction_of_image_crop(image_fraction=(1/4, 1/16), image_resize=2048, patch_size=512, crop="random"),

    # Basic
    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=1),

    # Morphology (scaling is already included in RandomResizedCrop)
    alb.OneOf([
        alb.ElasticTransform(),
        alb.GridDistortion(),], p=0.25),

    # Blur or Noise
    alb.OneOf([
        alb.AdvancedBlur(p=.25),
        alb.GaussNoise()], p=0.25),

    # Includes Brightness, Contrast, Hue and Saturation
    alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),

    # Normalization
    alb.Normalize(),

])

tellez_transforms_train_eff_b4 = alb.Compose([
    create_fraction_of_image_crop(image_fraction=(1/4, 1/16), image_resize=2048, patch_size=384, crop="random"),

    # Basic
    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=1),

    # Morphology (scaling is already included in RandomResizedCrop)
    alb.OneOf([
        alb.ElasticTransform(),
        alb.GridDistortion(),], p=0.25),

    # Blur or Noise
    alb.OneOf([
        alb.AdvancedBlur(p=.25),
        alb.GaussNoise()], p=0.25),

    # Includes Brightness, Contrast, Hue and Saturation
    alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),

    # Normalization
    alb.Normalize(),

])



basic_transforms_train = alb.Compose([
    alb.RandomBrightnessContrast(p=0.2),
    create_fraction_of_image_crop(image_fraction=(1/4, 1/16), image_resize=2048, patch_size=512, crop="random"),

    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=1),
    alb.Normalize(),
])


basic_transform_train_512 = alb.Compose([
    alb.RandomBrightnessContrast(p=0.2),
    alb.augmentations.geometric.resize.SmallestMaxSize(max_size=512, interpolation=2, always_apply=False, p=1),
    alb.RandomCrop(width=512, height=512, p=1),
    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=1),
    alb.Normalize(),
])

tellez_transforms_train_512 = alb.Compose([

    alb.augmentations.geometric.resize.SmallestMaxSize(max_size=512, interpolation=2, always_apply=False, p=1),
    alb.RandomCrop(width=512, height=512, p=1),

    # Basic
    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=1),

    # Morphology (scaling is already included in RandomResizedCrop)
    alb.OneOf([
        alb.ElasticTransform(),
        alb.GridDistortion(),], p=0.25),

    # Blur or Noise
    alb.OneOf([
        alb.AdvancedBlur(p=.25),
        alb.GaussNoise()], p=0.25),

    # Includes Brightness, Contrast, Hue and Saturation
    alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),

    # Normalization
    alb.Normalize(),

])

basic_transform_micron_calibrated = alb.Compose([
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomCrop(width=512, height=512, p=1),
    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=1),
    alb.Normalize(),
])

tellez_transforms_micron_calibrated = alb.Compose([

    alb.RandomCrop(width=512, height=512, p=1),

    # Basic
    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=1),

    # Morphology (scaling is already included in RandomResizedCrop)
    alb.OneOf([
        alb.ElasticTransform(),
        alb.GridDistortion(),], p=0.25),

    # Blur or Noise
    alb.OneOf([
        alb.AdvancedBlur(p=.25),
        alb.GaussNoise()], p=0.25),

    # Includes Brightness, Contrast, Hue and Saturation
    alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),

    # Normalization
    alb.Normalize(),

])

# For val we dont use rescaling.
calibrated_transforms_val_test = alb.Compose([
    alb.CenterCrop(width=512, height=512, p=1),
    alb.Normalize(),
])

# For val we dont use rescaling.
normalize_only_transform = alb.Compose([
    alb.Normalize(),
])

# For val we dont use rescaling.
basic_transforms_val_test = alb.Compose([
    create_fraction_of_image_crop(image_fraction=(1/4, 1/4), image_resize=2048, patch_size=512, crop="center"),
    alb.Normalize(),
])

# For val we dont use rescaling.
effb4_transforms_val_test = alb.Compose([
    create_fraction_of_image_crop(image_fraction=(1/4, 1/4), image_resize=2048, patch_size=384, crop="center"),
    alb.Normalize(),
])

# For val we dont use rescaling.
effb4_random_crop_val_test = alb.Compose([
    create_fraction_of_image_crop(image_fraction=(1/4, 1/4), image_resize=2048, patch_size=384, crop="random"),
    alb.Normalize(),
])

basic_random_crop_val_test = alb.Compose([
    create_fraction_of_image_crop(image_fraction=(1/4, 1/4), image_resize=2048, patch_size=512, crop="random"),
    alb.Normalize(),
])

# For val we dont use rescaling.
basic_transforms_val_test_scaling512 = alb.Compose([
    alb.augmentations.geometric.resize.SmallestMaxSize(max_size=512, interpolation=2, always_apply=False, p=1),
    alb.CenterCrop(width=512, height=512, p=1),
    alb.Normalize(),
])

# For val we dont use rescaling.
basic_transforms_val_test_scaling1024 = alb.Compose([
    alb.augmentations.geometric.resize.SmallestMaxSize(max_size=1024, interpolation=2, always_apply=False, p=1),
    alb.CenterCrop(width=1024, height=1024, p=1),
    alb.Normalize(),
])

# For val we dont use rescaling.
basic_transforms_val_test_scaling2048 = alb.Compose([
    alb.augmentations.geometric.resize.SmallestMaxSize(max_size=2048, interpolation=2, always_apply=False, p=1),
    alb.CenterCrop(width=2048, height=2048, p=1),
    alb.Normalize(),
])

# # For val we dont use rescaling.
basic_transforms_val_test_colorpreserving = alb.Compose([
    alb.augmentations.geometric.resize.SmallestMaxSize(max_size=512, interpolation=2, always_apply=False, p=1),
    alb.CenterCrop(width=512, height=512, p=1),
])


AUGMENTATIONS = {"tellez_train": tellez_transforms_train, "tellez_train_eff_b4": tellez_transforms_train_eff_b4, "basic_train": basic_transforms_train,
                 "basic_train_512": basic_transform_train_512, "tellez_train_512": tellez_transforms_train_512, "basic_train_calibrated": basic_transform_micron_calibrated, "tellez_train_calibrated": tellez_transforms_micron_calibrated,
                 "calibrated_eval": calibrated_transforms_val_test, "normalize_only": normalize_only_transform,
                 "basic_eval": basic_transforms_val_test, "basic_eval_eff_b4": effb4_transforms_val_test, "basic_eval_512": basic_transforms_val_test_scaling512}
