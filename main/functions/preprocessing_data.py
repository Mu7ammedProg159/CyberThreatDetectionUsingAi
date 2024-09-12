from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
import pandas as pd

def preprocess_data(dataset):

    orig_data = dataset
    dataset_columns_set = set(dataset.columns.str.lower())
    expected_columns = ['timestamp', 'uid', 'src_ip', 'src_port', 'dest_ip', 'dest_port',
       'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
       'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history',
       'src_pkts', 'src_ip_bytes', 'dest_pkts', 'dest_ip_bytes',
       'tunnel_parents', 'label', 'detailed-label']

    expected_columns_set = set(map(str.lower, expected_columns))
        
    #datasets = [labelled_data, unlabeled_data]

    if(dataset_columns_set != expected_columns_set):
        return()
    

    dataset.drop(columns=['uid', 'local_orig', 'local_resp', 'missed_bytes', 'tunnel_parents'], inplace=True)

    dataset['timestamp'].astype(int)

    unique_ip = dataset.drop_duplicates(subset=['src_ip'])
    unique_ip = unique_ip.drop(columns=unique_ip.columns.difference(['src_ip']))


    dataset['src_ip'] = dataset['src_ip'].astype('category')
    dataset['src_ip'] = dataset['src_ip'].cat.codes


    unique_ip_catcode = dataset.drop_duplicates(subset=['src_ip'])
    unique_ip_catcode = unique_ip_catcode.drop(columns=unique_ip_catcode.columns.difference(['src_ip']))


    unique_ip_table = pd.DataFrame({
        'Org_Src_IP': unique_ip['src_ip'],
        'CatCode_Src_IP': unique_ip_catcode['src_ip']
    })


    dataset['proto'] = dataset['proto'].astype('category')
    dataset['proto'] = dataset['proto'].cat.codes


    dataset.proto.unique()
    cc= dataset['proto'].unique()
    cc.sort()

    dataset.drop_duplicates(subset=['proto'])

    sortCC = dataset['proto'].unique()
    sortCC.sort()
    unique_proto_table = pd.DataFrame({
        'Org_Proto': orig_data['proto'].unique(),
        'CatCode_Proto': sortCC
    })


    dataset['service'].unique()

    dataset['service'] = dataset['service'].astype('category')
    (dataset['service'].cat.codes).unique()

    dataset['service'] = dataset['service'].cat.codes

    unique_service_table = pd.DataFrame({
        'Org_Service': orig_data['service'].unique(),
        'CC_Service': dataset['service'].unique()
    })

    sorted_data = unique_service_table.sort_values(by='CC_Service')
    unique_service_table = sorted_data



    dataset.loc[dataset['duration'] == '-', 'duration'] = 0

    dataset['duration'].astype(float)

    dataset['duration'] = pd.to_numeric(dataset['duration'], errors='coerce')
    dataset['duration'] = dataset.duration.round()

    dataset['duration'] = dataset['duration'].astype(int)

    dataset.loc[dataset['orig_bytes'] == '-', 'orig_bytes'] = -1

    dataset.loc[dataset['resp_bytes'] == '-', 'resp_bytes'] = -1
    dataset.loc[dataset['conn_state'] == '-', 'conn_state'] = 0
    dataset.loc[dataset['history'] == '-', 'history'] = 0
    dataset.loc[dataset['src_pkts'] == '-', 'src_pkts'] = -1
    dataset.loc[dataset['src_ip_bytes'] == '-', 'src_ip_bytes'] = -1
    dataset.loc[dataset['dest_pkts'] == '-', 'dest_pkts'] = -1
    dataset.loc[dataset['dest_ip_bytes'] == '-', 'dest_ip_bytes'] = -1

    
    dataset['detailed-label'].unique()

    dataset['detailed-label'] = dataset['detailed-label'].astype('category')
    (dataset['detailed-label'].cat.codes).unique()
    dataset['detailed-label'] = dataset['detailed-label'].cat.codes

    unique_detailedLabel_table = pd.DataFrame({
        'Org_detailed-label': orig_data['detailed-label'].unique(),
        'CC_detailed-label': dataset['detailed-label'].unique()
    })

    sorted_detailedLabel = unique_detailedLabel_table.sort_values(by='CC_detailed-label')
    unique_detailedLabel_table = sorted_detailedLabel


    # Replace 'Benign' with 0
    dataset['label'] = dataset['label'].replace('Benign', 0)

    # Replace all other values with 1
    dataset['label'] = dataset['label'].apply(lambda x: 1 if x != 0 else 0)

    print(dataset.head()) #1 -> Abnormal, 0 -> Normal

    unique_dest_ip = dataset.drop_duplicates(subset=['dest_ip'])
    unique_dest_ip = unique_dest_ip.drop(columns=unique_dest_ip.columns.difference(['dest_ip']))


    dataset['dest_ip'] = dataset['dest_ip'].astype('category')
    dataset['dest_ip'] = dataset['dest_ip'].cat.codes

    
    unique_dest_ip_catcode = dataset.drop_duplicates(subset=['dest_ip'])
    unique_dest_ip_catcode = unique_dest_ip_catcode.drop(columns=unique_dest_ip_catcode.columns.difference(['dest_ip']))


    unique_dest_ip_table = pd.DataFrame({
        'Org_Dest_IP': unique_dest_ip['dest_ip'],
        'CatCode_Dest_IP': unique_dest_ip_catcode['dest_ip']
    })



    unique_dest_ip_table.loc[unique_dest_ip_table['Org_Dest_IP'] == '192.168.100.103', 'Org_Dest_IP']

    dataset['conn_state'].unique()

    dataset['conn_state'] = dataset['conn_state'].astype('category')
    (dataset['conn_state'].cat.codes).unique()

    dataset['conn_state'] = dataset['conn_state'].cat.codes

    unique_connstate_table = pd.DataFrame({
        'Org_conn_state': orig_data['conn_state'].unique(),
        'CC_conn_state': dataset['conn_state'].unique()
    })


    sorted_connstate = unique_connstate_table.sort_values(by='CC_conn_state')
    unique_connstate_table = sorted_connstate


    dataset['history'].unique()

    dataset['history'] = dataset['history'].astype('category')
    (dataset['history'].cat.codes).unique()

    dataset['history'] = dataset['history'].cat.codes

    unique_history_table = pd.DataFrame({
        'Org_history_state': orig_data['history'].unique(),
        'CC_history': dataset['history'].unique()
    })


    sorted_history = unique_history_table.sort_values(by='CC_history')
    unique_history_table = sorted_history



    dataset['timestamp']=dataset['timestamp'].astype(int)

    dataset

    dataset['orig_bytes'] = dataset['orig_bytes'].astype(int)

    dataset['resp_bytes'] = dataset['resp_bytes'].astype(int)
    dataset['label'] = dataset['label'].astype(int)

    tables = [unique_ip_table, unique_dest_ip_table, unique_proto_table, unique_service_table, unique_history_table, unique_connstate_table, unique_detailedLabel_table]

    return(dataset, tables)
    #dataset.to_csv('../RealDataset/Network Datasets/Labelled/Training/Preprocessed_Laballed_Dataset.csv', index=False)

