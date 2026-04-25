import argparse
import shutil
from pathlib import Path


def restructure(src: Path, dst_root: Path, move: bool = False) -> None:
    """
    把 {case}_0000_input.nii.gz 与 {case}_0000_pred.nii.gz
    复制／移动到 dst_root/{case}/axial/ 下，并将 pred 重命名为 seg_v1.nii.gz。

    Parameters
    ----------
    src : Path
        搜索文件的源目录
    dst_root : Path
        输出目录（不会修改 src 下原始文件）
    move : bool
        True 则移动文件；False 则复制文件
    """
    for input_file in src.glob("*_0000_input.nii.gz"):
        case = input_file.stem.split("_")[0]                    # 提取 case 名
        pred_file = src / f"{case}_0000_pred.nii.gz"            # 预测文件
        if not pred_file.exists():
            print(f"[WARN] 找不到 {pred_file.name}，跳过 {case}")
            continue

        target_dir = dst_root / case / "axial"
        target_dir.mkdir(parents=True, exist_ok=True)

        dst_input = target_dir / input_file.name                # input 保持原名
        dst_pred = target_dir / "seg_v1.nii.gz"                 # pred 改名

        if move:
            shutil.move(str(input_file), dst_input)
            shutil.move(str(pred_file), dst_pred)
        else:
            shutil.copy2(str(input_file), dst_input)
            shutil.copy2(str(pred_file), dst_pred)

        action = "Moved" if move else "Copied"
        print(f"[OK] {action} files for case '{case}' → {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 *_input.nii.gz / *_pred.nii.gz 整理至目标目录 {case}/axial/ 并重命名 pred → seg_v1.nii.gz"
    )
    parser.add_argument("src", help="源目录（包含 *_input.nii.gz / *_pred.nii.gz）")
    parser.add_argument("dest", help="目标目录（整理后的输出位置）")
    parser.add_argument(
        "--move", action="store_true",
        help="移动文件（默认：复制文件，保留源文件）"
    )
    args = parser.parse_args()

    restructure(
        Path(args.src).expanduser().resolve(),
        Path(args.dest).expanduser().resolve(),
        move=args.move
    )


if __name__ == "__main__":
    main()