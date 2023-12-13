from server import app
from waitress import serve


def main():
    # app.run(host='::', port=5000, debug=True)
    serve(app, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
