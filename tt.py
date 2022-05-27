import base64
with open('11.txt', 'rb') as f:
    content = f.read()
    content = base64.b64encode(content)
    print(content)
    print(type(content))
    print(base64.b64decode(content).decode('utf-8'))