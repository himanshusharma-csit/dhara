from Config import confs
from Utils.Identifier import Identifier as id
from Utils.Dataset import load_preprocessed_dataset
from Utils.ProjectPaths import generate_path_instance
from Utils.Pipeline import initialize_captioning_model, evaluate_predictions
from Utils.FeatureExtractors import get_image_features_shape

# ============================================================
# SET THE EPOCH NUMBER TO LOAD
# ============================================================
TEST_EPOCH = 8
# ============================================================

paths = generate_path_instance(project_name=confs[id.project_directory],
                               dataset_choice=confs[id.dataset],
                               feature_directory=confs[id.feature_directory],
                               word_embedding_choice=confs[id.word_embeddings])

dataset = load_preprocessed_dataset(meta_paths=paths, dataset_choice=confs[id.dataset], finetuning=False)
dataset.batch_dataset(batch_size=confs[id.batch_size])

image_feature_shape = get_image_features_shape(input_shape=confs[id.feature_shape],
                                               extractor_choice=confs[id.feature_extractor],
                                               spatial=confs.get(id.use_patch_tokens, False))

model = initialize_captioning_model(paths=paths,
                                    batch_size=confs[id.batch_size],
                                    image_feature_shape=image_feature_shape,
                                    embedding_size=confs[id.embedding_size],
                                    dataset=dataset,
                                    word_embedding_type=confs[id.word_embeddings],
                                    decoder_type=confs[id.decoder_type],
                                    fusion_type=confs.get(id.multimodal_fusion))

if not model.state_exists(epoch=TEST_EPOCH, batch=0):
    raise FileNotFoundError(f"No saved state found for Epoch {TEST_EPOCH}.")

model.load_states(epoch=TEST_EPOCH, batch=0)
print(f"\nLoaded model state - Epoch: {TEST_EPOCH}")

model_predictions = model.validation(paths=paths,
                                     image_names=dataset.testing_image_names,
                                     batch_size=128,
                                     beam_size=5)

result, scores = evaluate_predictions(predictions=model_predictions,
                                      references=dataset.testing_image_captions)
teacher_forced_loss = model.evaluate_split_loss(paths=paths,
                                                image_captions=dataset.testing_image_captions,
                                                batch_size=128)

print(f"\nTest Results - Epoch {TEST_EPOCH}:")
print(f"Teacher-forced test loss:: {teacher_forced_loss}")
print(result)

model.save_test_results(epoch=TEST_EPOCH, batch=0, scores=scores,
                        teacher_forced_loss=teacher_forced_loss)
model.save_test_captions(epoch=TEST_EPOCH, batch=0, captions=model_predictions)

print(f"\nTest results saved to: {paths.test_results_dir()}")
