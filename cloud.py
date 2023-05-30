import os

import numpy as np
import requests
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_custom_trained_model_sample]
from typing import Dict, List, Union

from google.cloud import aiplatform
# from google.cloud.aiplatform_v1beta1 import Value
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if type(instances) == list else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # print("response")
    # print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    return response.predictions


with open("./0525_data/recording_1685001532559_13.pcm", 'rb') as f:
    pcm_data = np.frombuffer(f.read(), dtype=np.int16).tolist()
preds = predict_custom_trained_model_sample(
    project="635622715090",
    endpoint_id="2616340694851125248",
    location="us-central1",
    instances={ "pcm": pcm_data }
)

print(preds)
# [END aiplatform_predict_custom_trained_model_sample]
# r = requests.post("http://localhost:8080/isalive")
# with open("./0525_data/recording_1685001532559_13.pcm", 'rb') as f:
#     pcm_data = np.frombuffer(f.read(), dtype=np.int16).tolist()
# r = requests.post("http://localhost:8080/predict",json={
#     "instances":[{"pcm":pcm_data}]
# })
# print(r)
# r_json = r.json()
# print(f"the room number is :{r_json['predictions'][0]}")