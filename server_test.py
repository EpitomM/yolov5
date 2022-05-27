from flask import Flask, request

app = Flask(__name__)

@app.route('/hello', methods=["POST"])
def index():
    username= request.form.get('username')
    print('username=', username)
    # 逻辑判断
    msg = {"code": 200, 'msg': 'success'}
    return msg

if __name__ == '__main__':
    app.run(host='0.0.0.0',
      port=8081,
      debug=True)