import kglab
import torch
from torch_geometric.data import InMemoryDataset, HeteroData, Data
import os
from pathlib import Path

"""
!!!
Builds a custom dataset in PyG from an RDF based Knowledge Graph using SPARQL queries.
!!!
"""


class DatasetPyGRDF(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetPyGRDF, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['kg-data.pt']

    def download(self):
        pass

    def process(self):
        data = HeteroData()
        node_types = self.get_node_types() #to get each class
        for node_type in node_types: 
            data[node_type].x = torch.randn(self.get_num_class_features(node_type))

        df_links_numbers = self.get_links_numbers()
        node_dict = self.create_dict_nodes(df_links_numbers)
        grouped_df = self.create_node_mapping(df_links_numbers, node_dict)
        df_unique_key_list = self.get_list_keys(grouped_df)
        final_het_data = self.get_edge_index(df_unique_key_list, grouped_df, data) #create edge index
        print("final hetero data",final_het_data)
        torch.save(final_het_data, self.processed_paths[0])
    
    @staticmethod
    def rdf_kg():
        namespaces = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "if-gnn": "http://www.testgnn.org/2022/if-gnn#"
            # "eq":  "http://bos.ch/i40/core/equipment#"
        }
        return kglab.KnowledgeGraph(namespaces = namespaces).load_rdf("data/raw/testgnn-data.ttl")

    def get_node_types(self):
        """
            Return a list with the name of classes
        """
        path = os.getcwd() + "/sparql-queries/node_types_classes.rq"
        sparql = Path(path).read_text()
        df = self.rdf_kg().query_as_df(sparql)
        return df['class_name'].to_list()

    def get_num_class_features(self, class_name):
        """
            Return the number of instances and the number of class features of a given class
            df["count"][0] -> num_instances
            df["dt"][0] -> num_node_features
        """
        path = os.getcwd() + "/sparql-queries/instances_class_datatype_properties_per_class.rq"
        sparql = Path(path).read_text()
        query = sparql.replace("input", class_name)
        df = self.rdf_kg().query_as_df(query)
        print("count=",df["count"][0], df["dt"][0])
        return df["count"][0], df["dt"][0]

    def get_links_numbers(self):
        """
            Return the number of instances and the number of class features of a given class
            data[':Class1', ':property', ':Class2'].edge_index = [2, Number]
        """
        path = os.getcwd() + "/sparql-queries/instance_per_object_property.rq"
        sparql = Path(path).read_text()
        df = self.rdf_kg().query_as_df(sparql)
        return df

    def create_dict_nodes(self, df):
        """
            Create mapping
        """
        src_node = set(val for val in df['subject'])
        dst_node = set(val for val in df['object'])
        nodes = sorted(list(src_node.union(dst_node)))
        
        #create dictionary for nodes
        nodes_dict = {node: i for i, node in enumerate(nodes)}
        return nodes_dict

    def create_node_mapping(self, df, nodes_dict):
        """
        :param df:
        :param nodes_dict:
        :return: grouped_df
        """
        df['subject'] = df['subject'].map(nodes_dict)
        df['object'] = df['object'].map(nodes_dict)
        grouped_df = df.groupby(['subject_class', 'object_property', 'object_class'])
        return grouped_df

    def get_list_keys(self, grouped_df):
        """
        create list of s,p,o as keys
        :param grouped_df:
        :return:
        """
        grp = grouped_df.groups
        key_list = grp.keys()
        keylist = []
        keylist.extend(iter(key_list))
        return keylist

    def create_edge_index(self, s, p, o, src_class, dst_class, het_data):
        """
        func to create tensor of edge index
        :param s:
        :param p:
        :param o:
        :param src_class:
        :param dst_class:
        :param hetro_data:
        :return:
        """
        het_data[s, p, o].edge_index = torch.LongTensor([src_class, dst_class])
        # het_data[s, p, o].edge_label = torch.ones(159,dtype=torch.long).view(159,-1)
        print("final tensor",torch.LongTensor([src_class, dst_class]))
        return het_data

    def get_edge_index(self, key_list, grouped_df, het_data):
        """
        create func to create relation based on s,p,o
        :param key_list:
        :param grouped_df:
        :param het_data:
        :return:
        """
        for key, item in grouped_df:
            print(f"key={key}:item={item}")
            src_class = []
            dst_class = []
            for row_index, row in item.iterrows():
                if key_list[0] == key:
                    src, dst = row['subject'], row['object']
                    src_class.append(src)
                    dst_class.append(dst)
                else:
                    break
            data = self.create_edge_index(row['subject_class'], row['object_property'], row['object_class'], src_class,
                                          dst_class, het_data)
            key_list.remove(key)
        return data