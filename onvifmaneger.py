import socket
from wsdiscovery.discovery import ThreadedWSDiscovery as WSDiscovery
from onvif import ONVIFCamera
from urllib.parse import urlparse
import logging

logging.getLogger('zeep').setLevel(logging.CRITICAL)

def discover_onvif_devices(ip_base='192.168.1'):
    print("Scanning network for ONVIF cameras...")
    cameras = []
    

    for i in range(1, 255):
        ip = f"{ip_base}.{i}"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, 80))  # HTTP port for ONVIF
            if result == 0:
                print(f"Camera found at {ip}:80")
                cameras.append({'ip': ip, 'port': 80})
            sock.close()
        except Exception as e:
            print(f"Error scanning {ip}: {e}")
            continue

    print(f"Discovery complete. Found {len(cameras)} cameras.")
    return cameras

def get_rtsp_url(ip, port, username, password):
    try:
        print(f"Connecting to {ip}:{port}...")
        cam = ONVIFCamera(ip, port, username, password,wsdl_dir='./wsdl')

        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()

        if not profiles:
            print(f"No media profiles found on {ip}")
            return None

        profile_token = profiles[0].token
        stream_setup = {
            'Stream': 'RTP-Unicast',
            'Transport': {'Protocol': 'RTSP'}
        }

        # Combine StreamSetup and ProfileToken into a single dictionary
        request_params = {
            'StreamSetup': stream_setup,
            'ProfileToken': profile_token
        }

        uri_response = media_service.GetStreamUri(request_params)
      
        return uri_response.Uri

    except Exception as e:
        print(f"Error connecting to {ip}: {e}")
        return None


if __name__ == "__main__":
    # cameras = discover_onvif_devices()
    rtsp=get_rtsp_url('192.168.1.89',2000,'admin','admin')
    print(rtsp)
    # if not cameras:
    #     print("❌ No ONVIF cameras found on the network.")


