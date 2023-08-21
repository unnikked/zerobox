from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

if __name__ == '__main__':
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor.save_pretrained('./processor')
    model.save_pretrained('./model')