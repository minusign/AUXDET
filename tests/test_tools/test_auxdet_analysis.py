"""CPU regression tests for offline ablation analysis and CLI integration."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools/auxdet_tools"
sys.path.insert(0, str(TOOLS))
import eval_common as common


def script(name):
    spec = importlib.util.spec_from_file_location("test_" + name[:2], TOOLS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIZE = script("09_target_size_analysis.py")
COMPARE = script("10_compare_predictions.py")
PROFILE = script("11_profile_models.py")
SUMMARY = script("12_summarize_results.py")
DOMAIN = script("06_ap50_recall_by_view_band.py")


def image(gt=(), boxes=(), scores=(), ignored=None):
    return common.ImageRecord(
        "id", Path("unused.png"), 32, 32, "Space", "NIR",
        np.asarray(gt, np.float32).reshape(-1, 4), np.zeros(len(gt), np.int64),
        np.asarray(ignored if ignored is not None else [False] * len(gt), bool),
        np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
        np.zeros(len(boxes), np.int64))


def official_match(record):
    normal = record.gt_boxes[~record.gt_ignore]
    ignored = record.gt_boxes[record.gt_ignore]
    dets = record.predictions(1)[0]
    tp, fp = common.official_voc().tpfp_default(
        dets, normal, ignored, iou_thr=.5, use_legacy_coordinate=True)
    matches, _ = common.match_image(record)
    np.testing.assert_array_equal(tp[0], [r["status"] == "TP" for r in matches])
    np.testing.assert_array_equal(fp[0], [r["status"] == "FP" for r in matches])
    return [r["status"] for r in matches]


@pytest.mark.parametrize("record,expected", [
    (image([[0, 0, 4, 4]], [[0, 0, 4, 4]] * 2, [.7, .9]), ["FP", "TP"]),
    (image([[0, 0, 4, 4], [1, 0, 5, 4]], [[0, 0, 4, 4]] * 2, [.9, .8]), ["TP", "FP"]),
    (image([[0, 0, 4, 4]], [[0, 0, 4, 4]] * 2, [.9, .8], [True]), ["ignored", "ignored"]),
    (image([[1, 0, 5, 4], [0, 0, 4, 4]], [[0, 0, 4, 4]], [.9], [False, True]), ["ignored"]),
    (image([], [[0, 0, 1, 1]], [.5]), ["FP"]),
    (image([[0, 0, 4, 4]]), []),
    (image(), []),
    (image([[0, 0, 1, 0]], [[0, 0, 0, 0]], [.5]), ["TP"]),  # inclusive IoU == 0.5
])
def test_official_matching_edges(record, expected):
    assert official_match(record) == expected


def test_randomized_matching_and_ties():
    rng = np.random.default_rng(191)
    for _ in range(40):
        starts = rng.integers(0, 10, size=(10, 2))
        gt = np.concatenate((starts, starts+rng.integers(0, 8, size=(10, 2))), axis=1)
        starts = rng.integers(0, 10, size=(25, 2))
        boxes = np.concatenate((starts, starts+rng.integers(0, 8, size=(25, 2))), axis=1)
        official_match(image(gt, boxes, rng.choice([.2, .5, .9], 25), rng.random(10) < .3))


def test_threshold_curves_group_ties_and_ignore_difficult():
    record = image([[0, 0, 4, 4], [10, 10, 14, 14]],
                   [[0, 0, 4, 4], [20, 20, 24, 24], [10, 10, 14, 14]],
                   [.5, .5, .9], [False, True])
    curve = common.curves([record])
    assert len(curve) == 3
    assert curve[-1]["FPPI"] == 1
    assert curve[-1]["precision"] == .5
    assert curve[-1]["Recall"] == 1


@pytest.fixture
def fixture_data(tmp_path):
    base = tmp_path / "data/VOC2007"
    for directory in ("Annotations", "PNGImages", "ImageSets/Main"):
        (base / directory).mkdir(parents=True)
    definitions = {
        "0001": [([2, 2, 2, 2], 0)], "0002": [([2, 2, 4, 4], 0)],
        "0003": [([2, 2, 9, 9], 0)], "0004": [([2, 2, 16, 16], 0)],
        "0005": [([2, 2, 3, 3], 0), ([10, 10, 14, 14], 1)],
        "0006": [([2, 2, 6, 6], 0)], "0007": [], "0008": [([2, 2, 10, 10], 0)]}
    for key, boxes in definitions.items():
        view, band = ("Air", "LWIR") if key in ("0002", "0006") else ("Space", "NIR")
        objects = "".join(
            "<object><name>Target</name><difficult>" + str(ignore) + "</difficult><bndbox>" +
            "".join(f"<{k}>{v}</{k}>" for k, v in zip(("xmin", "ymin", "xmax", "ymax"), box)) +
            "</bndbox></object>" for box, ignore in boxes)
        (base / "Annotations" / f"{key}.xml").write_text(
            f"<annotation><size><width>32</width><height>32</height></size>"
            f"<view>{view}</view><band_type>{band}</band_type>{objects}</annotation>")
        Image.new("RGB", (32, 32), (20, 30, 40)).save(base / "PNGImages" / f"{key}.png")
    (base / "ImageSets/Main/train.txt").write_text("0001\n0002\n0003\n0004\n")
    (base / "ImageSets/Main/val.txt").write_text("0005\n0006\n0007\n0008\n")
    pipeline = [dict(type="LoadImageFromFile"), dict(type="Resize", scale=(64, 64), keep_ratio=True),
                dict(type="LoadAnnotations", with_bbox=True), dict(type="PackDetInputs")]
    settings = dict(type="VSBWILDVOCDetDataset", data_root=str(tmp_path / "data"),
                    data_prefix=dict(sub_data_root="VOC2007"), img_subdir="PNGImages",
                    ann_subdir="Annotations", metainfo=dict(classes=("Target",)),
                    ann_file="VOC2007/ImageSets/Main/val.txt", pipeline=pipeline)
    cfg = dict(model=dict(type="FasterRCNN", backbone=dict(type="ResNet"), neck=dict(type="AuxFPN"),
                          test_cfg=dict(rcnn=dict(score_thr=.001))),
               train_dataloader=dict(batch_size=2, dataset={**settings, "ann_file":"VOC2007/ImageSets/Main/train.txt"}),
               val_dataloader=dict(batch_size=2, dataset=settings),
               test_dataloader=dict(batch_size=2, dataset=settings),
               test_evaluator=dict(type="VOCMetric", metric="mAP", eval_mode="area"),
               train_cfg=dict(max_epochs=12), optim_wrapper=dict(optimizer=dict(lr=.02)))
    configs, caches, checkpoints, logs = {}, {}, {}, {}
    for model in ("B0", "B4"):
        config = tmp_path / f"{model}.py"
        values = copy.deepcopy(cfg)
        values["model"]["neck"]["variant"] = model
        config.write_text("\n".join(f"{key} = {value!r}" for key, value in values.items()))
        configs[model] = config
        payload = []
        for key in ("0005", "0006", "0007", "0008"):
            if model == "B0":
                boxes = {"0005": [[2,2,3,3], [2,2,3,3], [10,10,14,14]], "0006": [],
                         "0007": [[20,20,24,24]], "0008": []}[key]
            else:
                boxes = {"0005": [[3,2,4,3]], "0006": [[2,2,6,6]], "0007": [],
                         "0008": [[20,20,24,24]]}[key]
            payload.append(dict(img_id=key, ori_shape=(32,32), img_shape=(64,64),
                                scale_factor=(2.,2.), view="Air" if key=="0006" else "Space",
                                band_type="LWIR" if key=="0006" else "NIR",
                                pred_instances=dict(bboxes=np.array(boxes, np.float32).reshape(-1,4),
                                                    scores=np.linspace(.9,.7,len(boxes),dtype=np.float32),
                                                    labels=np.zeros(len(boxes),np.int64))))
        path = tmp_path / f"{model}.pkl"
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
        caches[model] = path
        checkpoints[model] = tmp_path / f"{model}.pth"
        checkpoints[model].write_bytes(b"SYNTHETIC TEST CHECKPOINT " + model.encode())
        logs[model] = tmp_path / f"{model}.log"
        logs[model].write_text("synthetic test provenance; not a training result")
    return dict(root=tmp_path, configs=configs, caches=caches, checkpoints=checkpoints, logs=logs,
                split=base/"ImageSets/Main/val.txt")


def test_real_id_alignment_and_coordinate_convention(fixture_data):
    f = fixture_data
    data = common.load_evaluation(f["configs"]["B0"], f["caches"]["B0"])
    assert [r.image_id for r in data.images] == ["0005","0006","0007","0008"]
    assert data.images[0].gt_boxes[0].tolist() == [2,2,3,3]  # no minus-one
    overall = common.evaluate(data.images, data.classes)
    domain = [common.evaluate(images, data.classes) for scope, _, _, images in common.groups(data) if scope=="domain"]
    assert overall["gt_count"] == 3
    assert overall["tp"] == 1
    assert overall["fp"] == 2
    assert overall["ignored_prediction_count"] == 1
    for key in ("gt_count", "tp", "fp", "fn", "image_count"):
        assert sum(row[key] for row in domain) == overall[key]
    assert overall["AP50"] == pytest.approx(1/3)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "metadata", "resize", "nan"])
def test_cache_integrity_rejections(fixture_data, mutation):
    f = fixture_data
    payload = common.load_prediction_file(f["caches"]["B0"])
    if mutation == "missing":
        payload.pop()
    elif mutation == "duplicate":
        payload.append(payload[0])
    elif mutation == "extra":
        payload[-1]["img_id"] = "wrong"
    elif mutation == "metadata":
        del payload[0]["band_type"]
    elif mutation == "resize":
        payload[0]["scale_factor"] = (.5,.5)
    else:
        payload[0]["pred_instances"]["scores"][0] = np.nan
    with pytest.raises(ValueError):
        common.attach_predictions(common.load_dataset(f["configs"]["B0"]), payload)


@pytest.mark.parametrize("mutation", ["crop", "tta", "resized_gt", "random_scale", "invalid_scale"])
def test_unsupported_test_transforms_are_rejected(fixture_data, mutation):
    f = fixture_data
    cfg = common.Config.fromfile(str(f["configs"]["B0"]))
    pipeline = cfg.test_dataloader.dataset.pipeline
    if mutation == "crop":
        pipeline.insert(2, dict(type="RandomCrop", crop_size=(32, 32)))
    elif mutation == "tta":
        pipeline.insert(2, dict(type="TestTimeAug"))
    elif mutation == "resized_gt":
        pipeline[1], pipeline[2] = pipeline[2], pipeline[1]
    elif mutation == "random_scale":
        pipeline[1]["type"] = "RandomChoiceResize"
    else:
        pipeline[1]["scale"] = (0, 64)
    cfg.dump(str(f["configs"]["B0"]))
    with pytest.raises(ValueError, match="test pipeline|original-image GT|fixed positive"):
        common.load_dataset(f["configs"]["B0"])


def test_original_prediction_indices_survive_threshold(fixture_data):
    f = fixture_data
    payload = common.load_prediction_file(f["caches"]["B0"])
    payload[0]["pred_instances"]["scores"] = np.array([.1,.9,.8], np.float32)
    data = common.attach_predictions(common.load_dataset(f["configs"]["B0"]), payload, score_thr=.5)
    rows, hits = common.match_image(data.images[0])
    assert rows[0]["prediction_index"] == 1
    assert hits[0] == 1


def test_size_boundaries_and_group_sums(fixture_data):
    f = fixture_data
    train = common.load_dataset(f["configs"]["B0"], split="train")
    bins = SIZE.fit_bins(train, SIZE.distribution(train), None, [0,4,10])
    assert [common.bin_index(v,bins) for v in [0,3.999,4,9.999,10,1e8]] == [0,0,1,1,2,2]
    data = common.load_evaluation(f["configs"]["B0"], f["caches"]["B0"])
    rows = SIZE.size_metrics(data,bins,"B0",data)
    assert sum(r["gt_count"] for r in rows if r["scope"]=="overall")==3
    assert sum(r["tp"] for r in rows if r["scope"]=="overall")==1
    assert all(r["delta_Recall"] == 0 for r in rows if r["gt_count"])
    with pytest.raises(ValueError):
        SIZE.fit_bins(data,SIZE.distribution(data), [.5])


def test_prediction_comparison(fixture_data):
    f=fixture_data
    a,b=(common.load_evaluation(f["configs"][m],f["caches"][m]) for m in ("B0","B4"))
    per_image, per_gt, matches, cases=COMPARE.compare(a,b)
    assert {r["state"] for r in per_gt}=={"lost","gained","both_missed","ignored"}
    assert sum(r["delta_TP"] for r in per_image)==0
    assert any(r["error_type"]=="localization" for r in matches)
    b.metadata["protocol_sha256"]="bad"
    with pytest.raises(ValueError):
        COMPARE.compare(a,b)


def test_parity_does_not_invent_missing_recall(tmp_path):
    path=tmp_path/"official.json"
    common.write_json(path, {"pascal_voc/AP50": .333})
    report=DOMAIN.parity_report({"AP50":1/3,"Recall":1/3,"gt_count":3},path)
    assert report["status"]=="partial"
    assert report["checks"]["Recall"]["status"]=="unavailable"
    common.write_json(path, {"AP50":.333,"Recall":.333,"gt_count":4})
    assert DOMAIN.parity_report({"AP50":1/3,"Recall":1/3,"gt_count":3},path)["status"]=="fail"


def invoke(name, args, tmp_path):
    env={**os.environ, "MPLCONFIGDIR":str(tmp_path/"mpl")}
    result=subprocess.run([sys.executable,"-B",str(TOOLS/name),*map(str,args)],
                          cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode==0, result.stdout+"\n"+result.stderr
    return result


def test_cli_pipeline_and_summary(fixture_data):
    f=fixture_data
    root=f["root"]
    bins=root/"bins"
    invoke("09_target_size_analysis.py",["--config",f["configs"]["B0"],"--fit-bins",
                                       "--output-dir",bins],root)
    entries=[]
    for model in ("B0","B4"):
        output=root/model
        invoke("06_ap50_recall_by_view_band.py",["--config",f["configs"][model],
               "--predictions",f["caches"][model],"--checkpoint",f["checkpoints"][model],
               "--model",model,"--output-dir",output],root)
        invoke("09_target_size_analysis.py",["--config",f["configs"][model],
               "--predictions",f["caches"][model],"--model",model,
               "--bins",bins/"size_bins.json","--output-dir",output/"size",
               "--baseline-config",f["configs"]["B0"],"--baseline-predictions",f["caches"]["B0"]],root)
        entries.append(dict(experiment=model,model=model,phase="screening",config=str(f["configs"][model]),
                            checkpoint=str(f["checkpoints"][model]),epoch=12,seed=0,training_log=str(f["logs"][model]),
                            predictions=str(f["caches"][model]),split=str(f["split"]),
                            training_protocol="fixture",evaluation_protocol="voc_area",
                            eval_dir=str(output),size_dir=str(output/"size"),profile_dir=""))
    manifest=root/"manifest.csv"
    common.write_csv(manifest,entries,common.MANIFEST_FIELDS)
    invoke("10_compare_predictions.py",["--baseline-config",f["configs"]["B0"],
           "--baseline-predictions",f["caches"]["B0"],"--candidate-config",f["configs"]["B4"],
           "--candidate-predictions",f["caches"]["B4"],"--candidate-name","B4",
           "--output-dir",root/"compare","--render-max",2],root)
    assert len(list((root/"compare/cases").glob("*.png")))==2
    invoke("12_summarize_results.py",["--manifest",manifest,"--output-dir",root/"paper"],root)
    assert (root/"paper/figures/screening_overall.pdf").is_file()
    assert len(common.read_csv(root/"paper/overall_summary.csv"))==2
    runs=[SUMMARY.load_run(row) for row in common.read_manifest(manifest)]
    bad=copy.deepcopy(runs)
    bad[1]["training_signature"]="different"
    with pytest.raises(ValueError):
        SUMMARY.validate_runs(bad)
    repeated=[]
    for run in runs:
        for seed in (1,2,3):
            item=copy.deepcopy(run)
            item.update(phase="repeat",seed=seed,experiment=f"{run['model']}_{seed}")
            item["metadata"]["checkpoint"]["sha256"]=f"synthetic-{run['model']}-{seed}"
            repeated.append(item)
    SUMMARY.validate_runs(repeated)
    stats=SUMMARY.aggregate([{"model":"x","value":v} for v in (1,2,3)],["model"],["value"])
    assert stats[0]["value_mean"]==2 and stats[0]["value_std"]==1
    repeated[-1]["seed"]=2
    with pytest.raises(ValueError):
        SUMMARY.validate_runs(repeated)
    repeated[-1]["seed"]=3
    repeated[-1]["metadata"]["checkpoint"]["sha256"]=repeated[-2]["metadata"]["checkpoint"]["sha256"]
    with pytest.raises(ValueError,match="same checkpoint"):
        SUMMARY.validate_runs(repeated)
    f["checkpoints"]["B4"].write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError,match="checkpoint differs"):
        SUMMARY.load_run(common.read_manifest(manifest)[1])


def test_profile_statistics():
    stats=PROFILE.latency_statistics([.001,.002,.003],2)
    assert stats["latency_mean_ms"]==2
    assert stats["latency_std_ms"]==1
    assert stats["throughput_images_per_second"]==1000


def test_ignored_only_and_empty_overall():
    row=common.evaluate([image([[0,0,1,1]],[[0,0,1,1]],[.8],[True])],("Target",))
    assert row["gt_count"]==0 and row["fp"]==0
    assert row["ignored_prediction_count"]==1 and row["Recall"] is None
    row=common.evaluate([image([],[[0,0,1,1]],[.8])],("Target",))
    assert row["fp"]==1 and row["AP50"]==0


def test_torch_official_cache_roundtrip(fixture_data):
    import torch
    f=fixture_data
    payload=common.load_prediction_file(f["caches"]["B0"])
    for record in payload:
        record["pred_instances"]={key:torch.from_numpy(value) for key,value in record["pred_instances"].items()}
    path=f["root"]/"torch_dump.pkl"
    with path.open("wb") as stream:
        pickle.dump(payload,stream)
    data=common.load_evaluation(f["configs"]["B0"],path)
    assert common.evaluate(data.images,data.classes)["tp"]==1


def test_train_bin_fit_rejects_validation_ids(fixture_data):
    f=fixture_data
    with pytest.raises(ValueError,match="training split"):
        common.load_dataset(f["configs"]["B0"],split="train",id_list=f["split"])


def test_missing_image_and_duplicate_split(fixture_data):
    f=fixture_data
    f["split"].write_text("0005\n0005\n")
    with pytest.raises(ValueError,match="duplicate"):
        common.load_dataset(f["configs"]["B0"])
    f["split"].write_text("0005\n0006\n0007\n0008\n")
    (f["root"]/"data/VOC2007/PNGImages/0008.png").rename(f["root"]/"moved.png")
    with pytest.raises(FileNotFoundError):
        common.load_dataset(f["configs"]["B0"])


@pytest.mark.parametrize("name", [
    "06_ap50_recall_by_view_band.py", "09_target_size_analysis.py",
    "10_compare_predictions.py", "11_profile_models.py", "12_summarize_results.py"])
def test_cli_help(name,tmp_path):
    result=invoke(name,["--help"],tmp_path)
    assert "usage:" in result.stdout


def test_auxfpn_configuration_contract(tmp_path):
    source = """
import importlib.util
from pathlib import Path
path=Path('tools/analysis_tools/verify_nsfpn.py').resolve()
spec=importlib.util.spec_from_file_location('audit',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
_,_,aux,_=module.load_modules('reference')
base=dict(in_channels=[256,512,1024,2048],out_channels=256,num_outs=4)
for kwargs in [dict(lfp_cfg={}),dict(lfp_levels=(0,)),dict(sfs_cfg={}),dict(sfs_fusions=(0,))]:
    try:
        aux.AuxFPN(**base,**kwargs)
    except ValueError as error:
        assert 'must be provided together' in str(error)
    else:
        raise AssertionError('Incomplete config was accepted')
print('configuration guard PASS')
"""
    result=subprocess.run([sys.executable,"-B","-c",source],cwd=ROOT,text=True,capture_output=True,
                          env={**os.environ,"MPLCONFIGDIR":str(tmp_path/"mpl")})
    assert result.returncode==0,result.stderr
