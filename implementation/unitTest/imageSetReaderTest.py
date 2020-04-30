from datasetReader.compressed.renderedImageReader import ImageSetReader

reader = ImageSetReader("/media/saurabh/e56e40fb-030d-4f7f-9e63" +
                        "-42ed5f7f6c71/preprocessing_new/")
dataset, metadata = reader.get_multi_view_dataset("03948459", [0, 45, 90])
print(metadata)
with dataset:
    img = dataset["1a640c8dffc5d01b8fd30d65663cfd42", 45]
    img.show()

dataset, metadata = reader.get_single_view_dataset("03948459", 0)
print(metadata)
with dataset:
    img = dataset["1a640c8dffc5d01b8fd30d65663cfd42"]
    img.show()
