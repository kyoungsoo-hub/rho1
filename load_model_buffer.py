from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.buffers.buffers import DictBuffer

model = FBcprModel.from_pretrained("facebook/metamotivo-M-1")

from huggingface_hub import hf_hub_download
import h5py

local_dir = "metamotivo-M-1-datasets"
dataset = "buffer_inference_500000.hdf5"  # a smaller buffer that can be used for reward inference
# dataset = "buffer.hdf5"  # the full training buffer of the model
buffer_path = hf_hub_download(
        repo_id="facebook/metamotivo-M-1",
        filename=f"data/{dataset}",
        repo_type="model",
        local_dir=local_dir,
    )
hf = h5py.File(buffer_path, "r")
print(hf.keys())

# create a DictBuffer object that can be used for sampling
data = {k: v[:] for k, v in hf.items()}
buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
buffer.extend(data)
print(buffer.shape())
