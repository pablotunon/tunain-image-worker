
import boto3
import fire
import json
import requests
import logging
import os

from botocore.exceptions import ClientError
from diffusers import DiffusionPipeline
import torch
import time


# Create SQS client
SQS_ENDPOINT = os.getenv('SQS_ENDPOINT', 'http://sqs.eu-west-1.localhost.localstack.cloud:4566')
IMAGE_QUEUE_URL = os.getenv('SQS_IMAGE_TASK_QUEUE_URL', 'http://sqs.eu-west-1.localhost.localstack.cloud:4566/000000000000/image-tasks')

sqs = boto3.client('sqs', endpoint_url=SQS_ENDPOINT)


s3_client = boto3.client('s3', endpoint_url='http://s3.localhost.localstack.cloud:4566')
bucket_name = 'image-generations'
resource_url = f'http://{bucket_name}.s3.localhost.localstack.cloud:4566/'

backend_url = 'http://127.0.0.1:8000/write-page'

USE_REFINER = False


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        return resource_url + object_name
    except ClientError as e:
        logging.error(e)
        return False

def send_results(page_id, output_image):
    timestamp = str(time.time()).replace(".", "")
    file_name = f'generations/{timestamp}_{page_id}.jpg'
    output_image.save(file_name)

    print('image generated, uploading file')
    image_url = upload_file(file_name, bucket_name)

    print(image_url)
    requests.post(backend_url, {
        "page_id": page_id,
        "image_url": image_url
    })

def process_with_refiner(image_prompt, base, refiner):
    n_steps = 40
    high_noise_frac = 0.8

    # run both experts
    image = base(
        prompt=image_prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    return refiner(
        prompt=image_prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

def process_without_refiner(image_prompt, base):
    return base(prompt=image_prompt).images[0]

def process_message(message, base, refiner):
    print(f"Received message: {message}")
    image_prompt = message['Body']
    # book_id = message['MessageAttributes']['BookId']['StringValue']
    page_id = message['MessageAttributes']['PageId']['StringValue']

    # Check format
    if USE_REFINER:
        output_image = process_with_refiner(image_prompt, base, refiner)
    else:
        output_image = process_without_refiner(image_prompt, base)

    send_results(page_id, output_image)


def listen_for_messages(base, refiner):
    while True:
        response = sqs.receive_message(
            QueueUrl=IMAGE_QUEUE_URL,
            AttributeNames=['All'],
            MaxNumberOfMessages=1,
            MessageAttributeNames=['All'],
            VisibilityTimeout=0,
            WaitTimeSeconds=20  # Adjust the wait time as needed
        )

        if 'Messages' in response and response['Messages']:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            
            process_message(message, base, refiner)
            
            # Delete the received message from the queue
            sqs.delete_message(
                QueueUrl=IMAGE_QUEUE_URL,
                ReceiptHandle=receipt_handle
            )

def main(
    # ckpt_dir: str = "../models/llama/llama-2-7b-chat",
    # tokenizer_path: str = "../models/llama/tokenizer.model",
    # temperature: float = 0.2,
    # top_p: float = 0.95,
    # max_seq_len: int = 2048,
    # max_batch_size: int = 8,
    # port: int = 8000,
):

    if USE_REFINER:
        base = DiffusionPipeline.from_pretrained(
            "../models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        base.to("cuda")
        refiner = DiffusionPipeline.from_pretrained(
            "../models/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.to("cuda")

    else:
        base = DiffusionPipeline.from_pretrained("../models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        base.to("cuda")
        refiner = None

    listen_for_messages(base, refiner)


if __name__ == "__main__":
    fire.Fire(main)
