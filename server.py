import os
import struct
import socket


# Constants
HOST = '0.0.0.0'  # Allow connections from any client
PORT = 50007
ENCODED_DIR = os.path.join(os.getcwd(), "encoded")
num_layers = 32


# send_all: Sends all bytes to the socket
def send_all(sock, data: bytes):
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Server listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)

            doc_id_data = conn.recv(8)
            if len(doc_id_data) < 8:
                print("Did not receive doc_id properly.")
                return
            doc_id = struct.unpack('>Q', doc_id_data)[0]  # unsigned long long big-endian

            for layer in range(num_layers):
                file_path = os.path.join(ENCODED_DIR, f"{doc_id}_layer_{layer}.pkl")
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    return

            send_all(conn, struct.pack('>Q', num_layers))

            for layer in range(num_layers):
                file_name = f"{doc_id}_layer_{layer}.pkl".encode('utf-8')
                file_path = os.path.join(ENCODED_DIR, f"{doc_id}_layer_{layer}.pkl")

                send_all(conn, struct.pack('>Q', len(file_name)))
                send_all(conn, file_name)

                with open(file_path, "rb") as f:
                    content = f.read()
                send_all(conn, struct.pack('>Q', len(content)))
                send_all(conn, content)

            print("All layers sent.")


if __name__ == "__main__":
    main()
