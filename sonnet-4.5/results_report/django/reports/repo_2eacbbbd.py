import django
from django.conf import settings

settings.configure(
    DEBUG=True,
    SECRET_KEY='test-key',
    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
    INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
)
django.setup()

from django.views.generic import UpdateView, DeleteView
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=100)
    slug = models.SlugField()
    class Meta:
        app_label = 'test'

# Test ModelFormMixin via UpdateView
class ArticleUpdateView(UpdateView):
    model = Article
    fields = ['title']
    success_url = '/articles/{category}/{id}/'  # 'category' doesn't exist in Article

article = Article(id=1, title='Test', slug='test')
article.__dict__.update({'id': 1, 'title': 'Test', 'slug': 'test'})

update_view = ArticleUpdateView()
update_view.object = article

print("Testing ModelFormMixin.get_success_url() with missing format parameter:")
try:
    url = update_view.get_success_url()
    print(f"Success URL: {url}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Exception type: {type(e).__name__}")

# Test DeletionMixin via DeleteView
class ArticleDeleteView(DeleteView):
    model = Article
    success_url = '/articles/{section}/{pk}/'  # 'section' and 'pk' don't exist

delete_view = ArticleDeleteView()
delete_view.object = article

print("\nTesting DeletionMixin.get_success_url() with missing format parameters:")
try:
    url = delete_view.get_success_url()
    print(f"Success URL: {url}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Exception type: {type(e).__name__}")