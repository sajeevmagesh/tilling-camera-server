import tempfile
from keras import models
import boto3

# Creating the low level functional client
client = boto3.client(
    's3',
    aws_access_key_id = 'AKIA5H64GCH7LG5OMBOJ',
    aws_secret_access_key = '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',
    region_name = 'us-west-2'
)


# Create the S3 object
response_data = client.get_object(
    Bucket = 'camera-ml',
    Key = 'TillingModel.h5'
)

model_name='model.h5'
response_data=response_data['Body']
response_data=response_data.read()
#save byte file to temp storage
with tempfile.TemporaryDirectory() as tempdir:
    with open(f"{tempdir}/{model_name}", 'wb') as my_data_file:
        my_data_file.write(response_data)
        #load byte file from temp storage into variable
        gotten_model=models.load_model(f"{tempdir}/{model_name}")
print(gotten_model.summary())