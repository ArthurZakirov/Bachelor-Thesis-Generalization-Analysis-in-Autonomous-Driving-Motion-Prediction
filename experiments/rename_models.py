"""
Module to rename Trajectron++ and MANTRA models
"""
import os


def rename_all_models_in_dir(experiment_dir):
    current_dir = os.getcwd()
    os.chdir(experiment_dir)
    for dir_name in os.listdir(os.getcwd()):
        new_dir_name = rename_dir(dir_name)
        os.rename(dir_name, new_dir_name)
    os.chdir(current_dir)


def rename_dir(dir_name):
    domains = [
        "nuScenes_Boston",
        "nuScenes_Queenstown",
        "nuScenes_Onenorth",
        "lyft_level_5",
        "openDD",
        "domain_mix",
    ]
    for domain in domains:
        if domain in dir_name:
            return domain


def main():
    """
    remove model creation timestamp from dir_name
    """
    Trajectron_models_path = "Trajectron/models"

    robot_path = os.path.join(Trajectron_models_path, "robot")
    rename_all_models_in_dir(robot_path)

    ablation_path = os.path.join(Trajectron_models_path, "ablation")
    for model_version in os.listdir(ablation_path):
        model_version_path = os.path.join(ablation_path, model_version)
        rename_all_models_in_dir(model_version_path)

    verbesserung_path = os.path.join(Trajectron_models_path, "Verbesserung")
    for experiment_type in os.listdir(verbesserung_path):
        experiment_type_path = os.path.join(verbesserung_path, experiment_type)
        for model_version in os.listdir(experiment_type_path):
            model_version_path = os.path.join(experiment_type_path, model_version)
            rename_all_models_in_dir(model_version_path)

    for training_stage in ["training_ae", "training_IRM"]:
        MANTRA_models_path = os.path.join("Mantra/models", training_stage)

        MANTRA_path = os.path.join(MANTRA_models_path, "MANTRA")
        rename_all_models_in_dir(MANTRA_path)

        verbesserung_path = os.path.join(MANTRA_models_path, "Verbesserung")
        for experiment_type in os.listdir(verbesserung_path):
            experiment_type_path = os.path.join(verbesserung_path, experiment_type)
            for model_version in os.listdir(experiment_type_path):
                model_version_path = os.path.join(experiment_type_path, model_version)
                rename_all_models_in_dir(model_version_path)


if __name__ == "__main__":
    main()
