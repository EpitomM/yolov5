import requests

url = "http://127.0.0.1:8081/hello"
data = {
	"username": "ssm"
}
response = requests.post(url, data=data)
print("request:", response.text)