import os
import socket
import cv2
import struct
import sys
import numpy as np
import time

def extract_reserved_fields(bmp_data):
    reserved1, reserved2 = struct.unpack_from('<hh', bmp_data, 6)
    return reserved1, reserved2

def extract_spacing(bmp_data):
    spacing1, spacing2 = struct.unpack_from('<ii', bmp_data, 0x26)
    return spacing1, spacing2


def flush_socket_recv(sock):
    sock.setblocking(False)
    try:
        while sock.recv(1024): pass
    except BlockingIOError:
        pass
    finally:
        sock.setblocking(True)

def convert_to_32bit_bgra(bmp_data):
    width = struct.unpack('<i', bmp_data[18:22])[0]
    height = struct.unpack('<i', bmp_data[22:26])[0]
    data_offset = struct.unpack('<I', bmp_data[10:14])[0]
    if struct.unpack('<I', bmp_data[30:34])[0] != 0:  
        raise ValueError("Only uncompressed BMP supported")
    img_array = np.frombuffer(bmp_data[data_offset:], dtype=np.uint8)
    img_array = img_array.reshape((height, width, 4))  
    return np.flipud(img_array)

def load_bmp_files(folder):
    bmp_files = []
    print("Loading BMP files from:", folder)
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith('.bmp'):
            print("Skipping non-BMP file:", filename)
            continue
        try:
            with open(os.path.join(folder, filename), 'rb') as f:
                bmp_data = f.read()
                print(f"Processing file: {filename} ({len(bmp_data)} bytes)")
                bits_per_pixel = struct.unpack('<H', bmp_data[28:30])[0]
                if bits_per_pixel == 32:
                    img = convert_to_32bit_bgra(bmp_data)
                    bmp_files.append((img, bmp_data))  # сохраняем и массив, и байты
                    print(f"Loaded: {filename} ({img.shape[1]}x{img.shape[0]})")
        except Exception as e:
            print(f"Error in {filename}: {e}")
    return bmp_files

def form_full_packet(img_data, frame_count, res1=0, res2=0, spacing1=0, spacing2=0):
    b, g, r = img_data[..., 0], img_data[..., 1], img_data[..., 2]
    grayscale_data = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    flipped_data = np.flipud(grayscale_data)
    #pixel_data = grayscale_data.tobytes()
    pixel_data = flipped_data.tobytes()
    height, width = img_data.shape[:2]  
    bytes_per_pixel = 1  
    row_size = width * bytes_per_pixel
    size_image = row_size * height
    
    BMP_HEADER_SIZE = 138

    offset_data_in_bmp_header = BMP_HEADER_SIZE 

    bmp_file_size = offset_data_in_bmp_header + size_image

    header = struct.pack(
        '<H I h h I I i i H H I I i i I I I I I I I 16I',
        0x4D42,                # file_type ("BM")
        bmp_file_size,         
        res1, res2,                  # reserved1, reserved2 (int16_t)
        offset_data_in_bmp_header,  # offset_data
        124,                   # размер структуры
        width, -height,        # width, height 
        1, 8,                  # planes, bit_count (8-бит grayscale)
        0, size_image,                  # compression, size_image 
        spacing1, spacing2,                  # x_pixels_per_meter, y_pixels_per_meter
        0, 0,                # colors_used, colors_important
        0x00ff0000,            # red_mask
        0x0000ff00,            # green_mask
        0x000000ff,            # blue_mask
        0xff000000,            # alpha_mask
        0x73524742,            # color_space_type (sRGB)
        *([0] * 16)            
    )
    
    uz_block_size = BMP_HEADER_SIZE + len(pixel_data)

    uz_header = struct.pack(
        '<IIII',
        0x555a4844,        # HdrSign ('UZHD') - согласовано с клиентом
        uz_block_size,       # BlockSize (тело пакета)
        0x44424D50,         # PacketType ('DBMP')
        frame_count          # TickMarker (например, номер кадра)
    )
    #print(f"DEBUG: UZ_Header bytes: {binascii.hexlify(uz_header)}")  # Логируем
    #print(f"\n[{time.strftime('%H:%M:%S')}] Отправка кадра #{frame_count} ({width}x{height})")
    #print(f"Iтоговый размер пакета: {len(uz_header) + len(header) + len(pixel_data)} байт")
    #print(f"  - UZ_StreamHeader: {len(uz_header)} байт")
    #print(f"  - BMP Header: {len(header)} байт")
    #print(f"  - Данные изображения: {uz_block_size} байт")
    #print(f"Отправка: UZ_Header={len(uz_header)}b, BMP_Header={len(header)}b, SEP=4b, Data={len(pixel_data)}b")
    #print(res1, " ", res2)

    full_data = uz_header + header  + pixel_data
    print("X - ",  height)
    print("Y - ",  width)

    return full_data 

def run_server_console(image_folder, fps):

    bmp_files = load_bmp_files(image_folder) #Загрузка изображений
    if not bmp_files:
        print("No BMP files found!")
        return

    frame_count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # создание сокета соединения
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 3456))
        s.listen(1)
        print("Server started on port 3456...")

        while True: #Ожидание подключения клиента
            try:
                conn, addr = s.accept()
                print(f"Client connected: {addr}")

                command = b""
                while not command.endswith(b'}'): #`Ожидание команды от клиента
                    data = conn.recv(1)
                    if not data:
                        break
                    command += data
                if command.endswith(b'}'):
                    print(f"Command received: {command.decode()}")
                    time.sleep(0.1)
                    flush_socket_recv(conn)
                else:
                    print("Invalid command received!")
                    conn.close()
                    continue

                while True: #Отправка изображений клиенту
                    img_data, bmp_data = bmp_files[frame_count % len(bmp_files)]
                    spacing1, spacing2 = extract_spacing(bmp_data)
                    res1, res2 = extract_reserved_fields(bmp_data)
                    print(res1," ", res2," ", spacing1," ", spacing2)
                    full_packet = form_full_packet(img_data, frame_count, res1, res2, spacing1, spacing2) #Формирование пакетов
                    try:
                        conn.sendall(full_packet)
                        frame_count += 1
                        time.sleep(1.0 / fps )
                    except Exception as e:
                        print(f"Клиент отключился: {e}")
                        conn.close()
                        break
            except Exception as e:
                print(f"Server error: {e}")

if __name__ == "__main__":
    IMAGE_FOLDER = "images"
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)  # Путь к .exe
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(exe_dir, 'images')

    print (f"Image folder: {image_folder}")

    FPS = 30
    run_server_console(image_folder, FPS)
