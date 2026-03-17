from app import create_app, find_free_port


if __name__ == "__main__":
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Open this in your browser: {url}")
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
