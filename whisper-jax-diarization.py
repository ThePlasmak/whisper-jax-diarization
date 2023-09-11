from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import time
from jax.experimental.compilation_cache import compilation_cache as cc
import os
import jax

start_time = time.time()

# environment variable
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# change from GPU to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Device: {jax.numpy.ones(3).device()}") # TFRT_CPU_0

# speed-up the compilation time if we close our kernel and want to compile the model again
cc.initialize_cache("./jax_cache")

# instantiate pipeline with float16 and enable batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2",
                            #   dtype=jnp.float16, # For most GPUs, the dtype should be set to jnp.float16. For A100 GPUs or TPUs, the dtype should be set to jnp.bfloat16
                            #   batch_size=1 # see https://github.com/sanchit-gandhi/whisper-jax#batching
                              )

# transcribe and return timestamps
outputs = pipeline("Animal Communication Ezekiel.m4a",  task="transcribe", return_timestamps=True)
text = outputs["text"]
chunks = outputs["chunks"]

# Time calculation
time_taken_seconds = time.time() - start_time # Calculate the time taken in seconds
time_taken_formatted = "{:.3f}".format(time_taken_seconds) # Format the time taken to three decimal places

# Calculate hours, minutes, and seconds
hours = int(time_taken_seconds // 3600)
minutes = int((time_taken_seconds % 3600) // 60)
seconds = int(time_taken_seconds % 60)

# Print the time taken in hours, minutes, and seconds format with three decimal places
print(
    "Time taken: {} hours, {} minutes, {} seconds".format(
        hours, minutes, time_taken_formatted
    )
)
