
from dataclasses import dataclass
from typing import List, Tuple, Union


DIFFUSION_T = 50


@dataclass(frozen=True)
class TaskInfo:
    name: str
    aliases: Tuple[str, ...]
    tmat: int
    tmat_fraction: float
    dataset_hint: str


TASKS: Tuple[TaskInfo, ...] = (
    TaskInfo("noisy", ("denoising", "noise", "gaussian-denoising"), 4, 0.08,
             "DIV2K/Flickr2K with synthetic Gaussian noise; CBSD68 for evaluation."),
    TaskInfo("rainy", ("deraining", "rain"), 8, 0.16,
             "Rain13K or Rain100H for paired deraining."),
    TaskInfo("jpeg", ("jpeg-compression", "jpeg-artifact", "car"), 12, 0.24,
             "DIV2K/Flickr2K with synthetic JPEG compression; LIVE1 for evaluation."),
    TaskInfo("snowy", ("desnowing", "snow"), 15, 0.30,
             "Snow100K for training and RealSnow for OOD evaluation."),
    TaskInfo("inpainting", ("paint", "mask"), 19, 0.38,
             "RePaint-style random mask protocol."),
    TaskInfo("raindrop", ("drop", "raindrop-removal"), 22, 0.44,
             "RainDrop paired raindrop-removal dataset."),
    TaskInfo("shadowed", ("shadow", "deshadow", "shadow-removal"), 27, 0.54,
             "SRD shadow removal dataset."),
    TaskInfo("lowlight", ("low-light", "low_light", "lowlight-enhancement"), 38, 0.76,
             "LOL-v1 for training and LOL-v2-real for OOD evaluation."),
    TaskInfo("hazy", ("dehazing", "haze"), 47, 0.94,
             "RESIDE/OTS for training and REVIDE for OOD evaluation."),
    TaskInfo("blurry", ("deblurring", "blur", "motion-blur"), 50, 1.00,
             "GoPro deblurring dataset."),
)


TASK_ORDER = [task.name for task in TASKS]


def canonical_task_name(name: str) -> str:
    key = name.strip().lower()
    for task in TASKS:
        if key == task.name or key in task.aliases:
            return task.name
    raise KeyError(f"Unknown DDL task: {name}. Available tasks: {', '.join(TASK_ORDER)}")


def tmat_for_task(name: str) -> int:
    canonical = canonical_task_name(name)
    for task in TASKS:
        if task.name == canonical:
            return task.tmat
    raise KeyError(name)


def selected_tasks(names: Union[str, List[str], Tuple[str, ...]]) -> List[TaskInfo]:
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]
    wanted = {canonical_task_name(name) for name in names}
    return [task for task in TASKS if task.name in wanted]


TMAT_TABLE = {
    alias: task.tmat
    for task in TASKS
    for alias in (task.name, *task.aliases)
}
