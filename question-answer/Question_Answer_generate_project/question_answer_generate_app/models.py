from django.db import models

# Create your models here.
class PDFUpload(models.Model):
    file = models.FileField(upload_to='pdfs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name
    
## create mode for  store vector an dasssitant detail
class VectorDetail(models.Model):
    vectorid=models.CharField(max_length=250)
    assistantid=models.CharField(max_length=250)
    created_at = models.DateTimeField(auto_now_add=True)
    

class FineTunedModel(models.Model):
    model_id = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.model_id