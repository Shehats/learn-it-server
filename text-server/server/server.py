from ..servicedef import text_pb2_grpc, text_pb2
from ..model import text_classification
import grpc
from concurrent import futures


class TextService(text_pb2_grpc.TextServiceServicer):
    def __init__(self, model_dict):
        super.__init__(self)
        self._model_dict = model_dict

    def GetRelatedWords(self, request_iterator, context):
        for req in request_iterator:
            _text = req.text
            _context = req.context
            pred, splitted = text_classification.predict(model_dict=self._model_dict, word_x=_text, word_y=_context)
            related_words = []

            for word, similarity in zip(splitted, pred):
                related_word = text_pb2.Releated(word=word, relation=similarity)
                related_words.append(related_word)
            
            yield text_pb2.TextResponse(relatedWords=related_words)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_dict = text_classification.create_model()
    text_pb2_grpc.add_TextServiceServicer_to_server(TextService(model_dict=model_dict), server)
    server.add_insecure_port('[::]:5005')
    server.start()

if __name__ == "__main__":
    serve()