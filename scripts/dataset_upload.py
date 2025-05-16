from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/research/d1/gds/kjshi/Surgical_Robot/surgical_data/splitted_datase.",
    repo_id="Onearth/USA-2K",
    repo_type="dataset",
)