

from insightface.app import FaceAnalysis
handler = FaceAnalysis('antelopev2', providers=[
                       'CUDAExecutionProvider', 'CPUExecutionProvider'],root='')
# # #CUDA?
handler.prepare(ctx_id=0, det_size=(640, 640))