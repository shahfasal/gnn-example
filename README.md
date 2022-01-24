# gnn-example
In this example we reproduce minimal example.    
This example comprises the following files    
1) Data an RDF file located at (data\raw\testgnn-data.ttl)
2) We create a dataset based on the RDF data executing SPARQL queries using the custom InMemoryDataset with the following code inside "datasetPyGRDF.py", the generated file is located at (data\processed\kg-data.pt)    
3) We execute the "model-test.py" file to generate the model on data created.    
4) We have included requirements.txt file in order to check the libraries we are using