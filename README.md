# Custom knowledge graph gnn-example (pytorch based)
This is minimal working example to load and train very basic model on the custom Knowledge Graph in pytorch-geometric.    
It is based on top of the example provided in pytorch-geometric [hetero_link_pred](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py) & a [discussion](https://github.com/pyg-team/pytorch_geometric/discussions/3221) with pytorch-geometric team.    
This example comprises the following files    
1) Data an RDF file located at (data\raw\testgnn-data.ttl)
2) We create a dataset based on the RDF data executing SPARQL queries using the custom InMemoryDataset with the following code inside "datasetPyGRDF.py", the generated file is located at (data\processed\kg-data.pt)    
3) We execute the "model-test.py" file to generate the model on data created.    
4) We have included requirements.txt file in order to check the libraries we are using

Note there can be other ways to achieve this but this approach suits our setup. Also please note in discussion with pyg team maintainer has asked to install pytorch-geometric from master <pip install git+https://github.com/pyg-team/pytorch_geometric.git>
