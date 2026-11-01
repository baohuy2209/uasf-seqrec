"""Regression tests for metadata handoff failures, using isolated fixture data."""
import json
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from metadata_pipeline import prepare
from metadata_pipeline.build_perturbations import process_dataset_perturbations
from metadata_pipeline.contracts import read_json, read_table, write_json
from metadata_pipeline.loader import load_catalog
from metadata_pipeline.report import generate_table1
from metadata_pipeline.validate import validate_dataset_artifacts
from taimma.data.split import compute_file_sha256, SequenceSample
from taimma.training.multimodal_trainer import MultimodalSequenceDataset
from taimma.models.sasrec import SASRecModel
from taimma.models.late_fusion import SourceSequentialEncoder, ResidualLateFusionModel

PERT = dict(missing_ratios=[0.0,0.8], seeds=[42,43], sources=["category","brand","title"])


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    raw = tmp_path/"raw"
    raw.mkdir()
    rows = [dict(reviewerID="U",asin=x,unixReviewTime=i,overall=5) for i,x in enumerate("ABCDEF")]
    (raw/"reviews_Beauty_5.json").write_text("\n".join(json.dumps(x) for x in rows),encoding="utf-8")
    metadata = [dict(asin=x,brand="ACME" if x != "B" else "",title="Title "+x if x != "C" else "",
                     categories=[["Beauty","Skin","Face"],["Beauty","Care"]] if x != "B" else []) for x in "ABCDEF"]
    (raw/"meta_Beauty.json").write_text("\n".join(json.dumps(x) for x in metadata),encoding="utf-8")
    config = tmp_path/"beauty.yaml"
    config.write_text(yaml.safe_dump(dict(dataset_name="Beauty_2014",min_interactions=5,max_seq_len=5,
                     expected_stats=dict(users=1,items=6,catalog_train_base_items=3,interactions=6))),encoding="utf-8")
    def fixture_vectors(titles, **kwargs):
        vectors = np.zeros((len(titles),8),dtype=np.float32)
        for i,title in enumerate(titles):
            if title:
                vectors[i,i] = 1
        return vectors,dict(encoder="test-fixture-only",mode="sentence-transformers",dim=8)
    monkeypatch.setattr(prepare,"get_title_embeddings",fixture_vectors)
    return raw, config, tmp_path/"artifacts"


@pytest.fixture
def handoff(inputs):
    raw, config, out = inputs
    prepare.process_dataset_metadata("Beauty",str(raw),str(out),str(config),"beauty")
    process_dataset_perturbations("beauty",out,PERT["missing_ratios"],PERT["seeds"],PERT["sources"])
    return out/"beauty", raw


def validate(path, **kwargs):
    return validate_dataset_artifacts(path,perturbation_config=PERT,**kwargs)


def update_hash(path, name):
    manifest = read_json(path/"manifest.json")
    manifest["artifact_sha256"][name] = compute_file_sha256(str(path/name))
    write_json(path/"manifest.json",manifest)


def test_encoder_failure_has_no_random_fallback(monkeypatch):
    stub = types.ModuleType("sentence_transformers")
    def fail(*args,**kwargs):
        raise RuntimeError("encoder unavailable")
    stub.SentenceTransformer = fail
    monkeypatch.setitem(sys.modules,"sentence_transformers",stub)
    with pytest.raises(RuntimeError,match="encoder unavailable"):
        prepare.get_title_embeddings(["real title"])


def test_missing_raw_metadata_stops_before_output(inputs):
    raw,config,out = inputs
    (raw/"meta_Beauty.json").unlink()
    with pytest.raises(FileNotFoundError,match="Metadata missing"):
        prepare.process_dataset_metadata("Beauty",str(raw),str(out),str(config),"beauty")
    assert not out.exists()


def test_wrong_statistics_stops_before_output(inputs):
    raw,config,out = inputs
    with pytest.raises(ValueError,match="statistic users"):
        prepare.process_dataset_metadata("Beauty",str(raw),str(out),str(config),"beauty",expected_stats={"users":999})
    assert not out.exists()


def test_complete_handoff_and_multilabel_raw_verification(handoff):
    path,raw = handoff
    result = validate(path,verify_raw_hash=True,raw_root=raw)
    assert result["status"] == "PASS" and result["provenance"] == "VERIFIED_SHA256_AND_RAW_MAPPING"
    assert json.loads(read_table(path/"metadata.csv").iloc[0].categories) == ["Skin","Face","Care"]
    attrs,masks = load_catalog(path)
    assert attrs["category"].shape == (3,3)
    sample = SequenceSample(0,[0,1,2],1,2)
    batch = MultimodalSequenceDataset([sample],attrs,masks)[0]
    assert batch["source_seqs"]["category"].shape == (3,3)
    model = ResidualLateFusionModel(SASRecModel(3,max_seq_len=3,embedding_dim=8,num_heads=1),
             {"category":SourceSequentialEncoder("category",3,feature_dim=8,max_seq_len=3,num_heads=1)},
             sources_enabled=("category",),p_drop=0)
    model.eval()
    with torch.no_grad():
        score = model.predict_all(torch.tensor([[0,1,2]]), {"category":batch["source_seqs"]["category"].unsqueeze(0)},
                      {"category":attrs["category"]},{"category":masks["category"]})
    assert torch.isfinite(score).all()


@pytest.mark.parametrize("name",["category_vocab.json","brand_vocab.json"])
def test_missing_vocab_fails(handoff,name):
    path,_ = handoff
    (path/name).unlink()
    with pytest.raises(FileNotFoundError):
        validate(path)


def test_wrong_availability_fails_even_with_updated_checksum(handoff):
    path,_ = handoff
    table = read_table(path/"availability_mask.csv")
    table["has_brand"] = False
    table.to_csv(path/"availability_mask.csv",index=False)
    update_hash(path,"availability_mask.csv")
    with pytest.raises(ValueError,match="availability/metadata mismatch"):
        validate(path)


def test_fake_embedding_mode_fails(handoff):
    path,_ = handoff
    manifest = read_json(path/"manifest.json")
    manifest["title_embedding"]["mode"] = "deterministic-hash-projection"
    write_json(path/"manifest.json",manifest)
    with pytest.raises(ValueError,match="Real Title encoder"):
        validate(path)


@pytest.mark.parametrize("kind",["missing_mask","wrong_mask","wrong_permutation"])
def test_corrupt_perturbations_fail(handoff,kind):
    path,_ = handoff
    mask = path/"perturbations/masks/mask_brand_ratio_0.80_seed_42.csv"
    if kind == "missing_mask":
        mask.unlink()
    elif kind == "wrong_mask":
        table = read_table(mask)
        table["simulated_available"] = True
        table.to_csv(mask,index=False)
    else:
        perm = path/"perturbations/permutations/perm_brand_seed_42.csv"
        table = read_table(perm)
        table["permuted_source_item_id"] = 1
        table.to_csv(perm,index=False)
    with pytest.raises((ValueError,FileNotFoundError)):
        validate(path)


def test_relocated_raw_sources_and_explicit_artifact_only(handoff,tmp_path):
    path,raw = handoff
    relocated = tmp_path/"moved"
    raw.rename(relocated)
    with pytest.raises(ValueError,match="Raw inputs missing"):
        validate(path)
    assert validate(path,raw_root=relocated,verify_raw_hash=True)["status"] == "PASS"
    assert validate(path,artifacts_only=True)["provenance"] == "NOT_CHECKED_ARTIFACTS_ONLY"
    with pytest.raises(ValueError,match="cannot verify"):
        validate(path,artifacts_only=True,verify_raw_hash=True)


def test_report_requires_all_sources_and_preserves_unknown(handoff,tmp_path):
    path,_ = handoff
    out = tmp_path/"table.csv"
    with pytest.raises(FileNotFoundError):
        generate_table1(str(path.parent),str(out))
    assert not out.exists()
    frame = generate_table1(str(path.parent),str(out),datasets=["beauty"])
    assert frame.users.tolist() == [1,1] and frame.interactions.tolist() == [6,6]
    cov = read_json(path/"coverage.json")
    del cov["train_base_catalog"]["brand"]
    write_json(path/"coverage.json",cov)
    with pytest.raises(KeyError):
        generate_table1(str(path.parent),str(tmp_path/"bad.csv"),datasets=["beauty"])


def test_failed_rebuild_preserves_previous_handoff(handoff,inputs,monkeypatch):
    path,_ = handoff
    before = (path/"metadata.csv").read_bytes()
    def fail(*args,**kwargs):
        raise RuntimeError("encode failed")
    monkeypatch.setattr(prepare,"get_title_embeddings",fail)
    raw,config,out = inputs
    with pytest.raises(RuntimeError,match="encode failed"):
        prepare.process_dataset_metadata("Beauty",str(raw),str(out),str(config),"beauty",overwrite=True)
    assert (path/"metadata.csv").read_bytes() == before


@pytest.mark.parametrize("source",["category","brand","title"])
@pytest.mark.parametrize("ratio",["0.00","0.80"])
def test_saved_perturbations_disable_candidate_scores(handoff,source,ratio):
    path,_ = handoff
    mask = path/f"perturbations/masks/mask_{source}_ratio_{ratio}_seed_42.csv"
    perm = path/f"perturbations/permutations/perm_{source}_seed_42.csv"
    attrs,masks = load_catalog(path,{source:mask},{source:perm})
    sample = SequenceSample(0,[0,1,2],1,2)
    loaded = MultimodalSequenceDataset([sample],attrs,masks)[0]
    vocab_size = attrs[source].shape[1] if source == "title" else int(attrs[source].max())
    encoder = SourceSequentialEncoder(source,vocab_size,feature_dim=8,max_seq_len=3,num_heads=1,
                                      is_dense_feature=source == "title")
    with torch.no_grad(): encoder.output_projection.weight.copy_(torch.eye(8))
    model = ResidualLateFusionModel(SASRecModel(3,max_seq_len=3,embedding_dim=8,num_heads=1),
              {source:encoder},sources_enabled=(source,),p_drop=0)
    model.eval()
    with torch.no_grad():
        joint,base,residual = model.compute_all_scores(torch.tensor([[0,1,2]]),
              {source:loaded["source_seqs"][source].unsqueeze(0)},
              {source:attrs[source]},{source:masks[source]},apply_dropout=False)
    assert torch.isfinite(joint).all()
    torch.testing.assert_close(joint-base,residual[source]*masks[source].unsqueeze(0),atol=1e-6,rtol=1e-6)
    assert torch.equal(joint[:,masks[source] == 0],base[:,masks[source] == 0])
    if ratio == "0.00": assert torch.count_nonzero(joint-base)>0
