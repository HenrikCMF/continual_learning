import fastavro
import json
import fastavro.schema
import pandas as pd
import numpy as np

def save_AVRO_default(df2, timestamps, schemapath, accuracy, path, original_size, codec='null'):

    with open(schemapath, "r") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(df2, pd.DataFrame):
            df2 = pd.DataFrame(df2)
            w,l = np.shape(df2)
    features=[]

    for i in df2:
         features.append(i)
    measurement_list=[]

    metadata= {"original_size": str(original_size)}
    for i in range(np.shape(df2)[0]):
        try:
            sensor_data= {f"{j}": round(df2[j][i], accuracy) for j in features}
        except:
            sensor_data= {f"{j}": df2[j][i] for j in features}
        sensor_data['timestamp']=timestamps[i]
        measurement_list.append(sensor_data)
    with open(path, "wb") as out:
        fastavro.writer(out, schema, measurement_list, codec=codec, metadata=metadata)


def save_AVRO_array(df2, timestamps, schemapath, accuracy, path, original_size, codec='null'):

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

    timestampspersample=int(len(timestamps)/np.shape(df2)[0])
    if len(timestamps)/np.shape(df2)[0]%2!=0:
         timestampspersample+=1
    for i in range(np.shape(df2)[0]):
        sensor_data= {f"{j}": round(df2[j][i], accuracy) for j in features}

        timestampindex=i*timestampspersample
        sensor_data['timestamp']=timestamps[timestampindex:timestampindex+timestampspersample]
        measurement_list.append(sensor_data)
    
    with open(path, "wb") as out:
        fastavro.writer(out, schema, measurement_list, codec=codec, metadata=metadata)

def load_AVRO_file(path):
    data_decompressed=[]
    metadata=0

    with open(path, "rb") as file:
        reader = fastavro.reader(file)
        schematype=reader.writer_schema
        metadata=int(reader.metadata['original_size'])
        
        for record in reader:
            data_decompressed.append(record)
    data_decompressed=pd.DataFrame(data_decompressed)
    timestamps=data_decompressed['timestamp'].values
    timestamplist=[]

    timestamps=pd.DataFrame(timestamps)
    data_decompressed=data_decompressed.drop(columns=['timestamp'])
    return data_decompressed,timestamps, schematype.get('name'), metadata
