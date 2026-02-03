import urllib.request
import zipfile

url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
print("Downloading Tiny ImageNet...")
urllib.request.urlretrieve(url, "tiny-imagenet-200.zip")
print("Download complete!")

print("Extracting...")
with zipfile.ZipFile("tiny-imagenet-200.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
print("Done!")
exit()