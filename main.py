import pandas as pd
from hclust import hclust
from graphs import display, histogram, instances_plot, map
from sklearn.decomposition import PCA
from geopandas import GeoDataFrame

table = pd.read_csv("imbatranirea_demografica.csv", index_col=0)

variables = list(table)[1:]

x = table[variables].values
hclust_model = hclust(table, variables)
print(hclust_model.h)
optim_partition = hclust_model.partition("Partitia optimala")
partition_table = pd.DataFrame(
    data={
        "Partitia optimala": optim_partition
    }, index=hclust_model.instances
)
partition_3 = hclust_model.partition("Partitia cu 3 clusteri", clusters_no=3)
partition_table["Partitia cu 3 clusteri"] = partition_3

print(partition_table)
partition_table.to_csv("partitii.csv")

for v in variables:
    histogram(table, v, optim_partition, title="Histograme partitie optimala - " + v)
    histogram(table, v, partition_3, title="Histograme partitie cu 3 clusteri - " + v)

pca = PCA(n_components=2)
x = table[variables].values
pca.fit(x)
z = pca.transform(x)
instances_plot(z, optim_partition, instances=hclust_model.instances, title="Plot partitie optimala in axe principale")
instances_plot(z, partition_3, instances=hclust_model.instances, title="Plot partitie 3 clusteri in axe principale")

t_shp = GeoDataFrame.from_file("RO_NUTS2/Ro.shp")
print(list(t_shp))
map(t_shp, "sj", partition_table)

display()
