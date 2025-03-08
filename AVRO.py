import fastavro
import json
import fastavro.schema
import pandas as pd
import numpy as np
#This file contains methods related to saving on the AVRO file format.

def save_AVRO_default(df2, timestamps, schemapath, accuracy, path, original_size, codec='null'):
    #This function is for saving samples which only needs 1 timestamp per measurement
    with open(schemapath, "r") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(df2, pd.DataFrame):
            df2 = pd.DataFrame(df2)
            w,l = np.shape(df2)
            if w<l:
                df2=np.transpose(df2) 
    features=[]
    #Get feature names
    for i in df2:
         features.append(i)
    measurement_list=[]
    #Save original size in metadata
    metadata= {"original_size": str(original_size)}
    #Save data in a list and write to a file
    for i in range(np.shape(df2)[0]):
        sensor_data= {f"{j}": round(df2[j][i], accuracy) for j in features}
        sensor_data['timestamp']=timestamps[i]
        measurement_list.append(sensor_data)
    with open(path, "wb") as out:
        fastavro.writer(out, schema, measurement_list, codec=codec, metadata=metadata)


def save_AVRO_array(df2, timestamps, schemapath, accuracy, path, original_size, codec='null'):
    #This function is for downsampled data, but where all timestamps are still needed
    with open(schemapath, "r") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(df2, pd.DataFrame):
            df2 = pd.DataFrame(df2)
            w,l = np.shape(df2)
            if w<l:
                df2=np.transpose(df2) 
    features=[]
    for i in df2:
         features.append(i)
    measurement_list=[]
    metadata= {"original_size": str(original_size)}
    #Based on length of timestamps compared to length of data
    #get how many timestamps need to be saved per iteration
    timestampspersample=int(len(timestamps)/np.shape(df2)[0])
    if len(timestamps)/np.shape(df2)[0]%2!=0:
         timestampspersample+=1
    for i in range(np.shape(df2)[0]):
        sensor_data= {f"{j}": round(df2[j][i], accuracy) for j in features}
        #Save a range of timestamps per iteration
        timestampindex=i*timestampspersample
        sensor_data['timestamp']=timestamps[timestampindex:timestampindex+timestampspersample]
        measurement_list.append(sensor_data)
    
    with open(path, "wb") as out:
        fastavro.writer(out, schema, measurement_list, codec=codec, metadata=metadata)


#Loads the compressed files
def load_AVRO_file(path):
    data_decompressed=[]
    metadata=0
    #Open the file and read with avro
    #Note down both schematype and metadata
    with open(path, "rb") as file:
        reader = fastavro.reader(file)
        schematype=reader.writer_schema
        metadata=int(reader.metadata['original_size'])
        
        for record in reader:
            data_decompressed.append(record)
    data_decompressed=pd.DataFrame(data_decompressed)
    timestamps=data_decompressed['timestamp'].values
    timestamplist=[]
    #If the timestamps were saved in arrays, unwrap them
    if schematype.get('name')!='Deflate':
        for i in range(len(timestamps)):
            timestamplist.extend(timestamps[i])
        timestamps=timestamplist
    timestamps=pd.DataFrame(timestamps)
    data_decompressed=data_decompressed.drop(columns=['timestamp'])
    return data_decompressed,timestamps, schematype.get('name'), metadata
