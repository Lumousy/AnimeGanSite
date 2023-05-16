from django.db import models


class User(models.Model):
    username = models.CharField(verbose_name="用户名", max_length=10)
    password = models.CharField(verbose_name="密码", max_length=10)


class Image(models.Model):
    user = models.ForeignKey(verbose_name="关联用户id", to="User", on_delete=models.CASCADE)
    input_img = models.ImageField(upload_to='input')
    output_img = models.ImageField(upload_to='output')


class Manager(models.Model):
    username = models.CharField(verbose_name="用户名", max_length=10)
    password = models.CharField(verbose_name="密码", max_length=10)
