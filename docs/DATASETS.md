# Datasets

DDL uses paired image restoration data in two settings: single-task OOD
generalization and multi-task unified restoration. This repository does not
redistribute datasets. Please download each dataset from the official project
or the authors' release page, and follow the corresponding license.

## Expected Folder Layout

For single-task fine-tuning:

```text
data/
  dehazing/
    train_input/
    train_gt/
    val_input/
    val_gt/
```

For multi-task unified learning:

```text
data/
  noisy/train_input/       noisy/train_gt/
  rainy/train_input/       rainy/train_gt/
  jpeg/train_input/        jpeg/train_gt/
  snowy/train_input/       snowy/train_gt/
  inpainting/train_input/  inpainting/train_gt/
  raindrop/train_input/    raindrop/train_gt/
  shadowed/train_input/    shadowed/train_gt/
  lowlight/train_input/    lowlight/train_gt/
  hazy/train_input/        hazy/train_gt/
  blurry/train_input/      blurry/train_gt/
```

The loader also accepts `input/` and `gt/` when `train_input/` and `train_gt/`
are absent.

## Matched Timesteps

| Task folder | Degradation | Matched timestep |
|---|---|---:|
| `noisy` | Gaussian noise | `0.08T` |
| `rainy` | Rain streaks | `0.16T` |
| `jpeg` | JPEG artifacts | `0.24T` |
| `snowy` | Snow | `0.30T` |
| `inpainting` | Masked holes | `0.38T` |
| `raindrop` | Raindrops | `0.44T` |
| `shadowed` | Shadows | `0.54T` |
| `lowlight` | Low-light | `0.76T` |
| `hazy` | Haze | `0.94T` |
| `blurry` | Motion blur | `T` |

The default code uses `T=50`, so the integer timesteps are
`4, 8, 12, 15, 19, 22, 27, 38, 47, 50`.

## Dataset Links

| Dataset / protocol | Used for | Link |
|---|---|---|
| DIV2K | clean GT images; denoising and JPEG synthesis | https://data.vision.ee.ethz.ch/cvl/DIV2K/ |
| Flickr2K | clean GT images; denoising and JPEG synthesis | https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar |
| CBSD68 | image denoising evaluation | https://github.com/cszn/DnCNN/tree/master/testsets/CBSD68 |
| LIVE1 | JPEG artifact evaluation | https://live.ece.utexas.edu/research/quality/subjective.htm |
| Rain100H / Rain13K | deraining training and evaluation | https://github.com/swz30/MPRNet |
| SPA+ / SPA-Data | real-world deraining OOD evaluation | https://github.com/stevewongv/SPANet |
| Snow100K | desnowing training | https://sites.google.com/view/yunfuliu/desnownet |
| RealSnow | real-world desnowing OOD evaluation | please follow the RealSnow release used by the corresponding desnowing benchmark |
| RePaint mask protocol | inpainting mask generation/evaluation protocol | https://github.com/andreas128/RePaint |
| RainDrop | raindrop removal | https://github.com/rui1996/DeRaindrop |
| SRD | shadow removal | https://github.com/stevewongv/InstanceShadowDetection |
| LOL-v1 | low-light enhancement training | https://daooshee.github.io/BMVC2018website/ |
| LOL-v2-real | real-world low-light OOD evaluation | https://github.com/caiyuanhao1998/Retinexformer |
| RESIDE / OTS | dehazing training | https://sites.google.com/view/reside-dehaze-datasets |
| REVIDE | real-world dehazing OOD evaluation | https://github.com/BookerDeWitt/REVIDE_Dataset |
| GoPro | image deblurring | https://seungjunnah.github.io/Datasets/gopro |

Some benchmarks are mirrored by many restoration projects. For reproducibility,
we recommend recording the exact split, preprocessing script, and checksum used
in each experiment.
