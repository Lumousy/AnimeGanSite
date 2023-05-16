import os

from django.conf import settings
from django.contrib.auth import logout
from django.http import HttpResponse, FileResponse, Http404
from django.shortcuts import render, redirect
from django.urls import reverse
import os
from PIL import Image
import torch

from login import models


def index(request):
    return render(request, 'index.html')


def login(request):
    if request.method == "GET":
        return render(request, 'login.html')

    else:
        username = request.POST.get('name')
        password = request.POST.get('psw')
        print(username)
        print(password)
        # 登录验证
        user_object = models.User.objects.filter(username=username, password=password).first()
        print(user_object)
        if user_object:
            request.session["info"] = {'name': user_object.username, 'id': user_object.id}
            return redirect('/home/')
        else:
            return render(request, 'login.html', {'error': '用户名或密码错误'})


def manage_login(request):
    if request.method == "GET":
        return render(request, 'manage_login.html')

    else:
        name = request.POST.get('m_name')
        psw = request.POST.get('m_psw')

        # 登录验证
        manege_object = models.Manager.objects.filter(username=name, password=psw).first()

        if manege_object:
            request.session["info"] = {'name': manege_object.username}
            return redirect('/user/list/')
        else:
            return render(request, 'manage_login.html', {'error': '用户名或密码错误'})


def enroll(request):
    if request.method == "GET":
        return render(request, 'enroll.html')

    else:
        username = request.POST.get('name')
        password = request.POST.get('psw')
        models.User.objects.create(username=username, password=password)
        return redirect('/index/')


def home(request):
    info_dict = request.session.get('info')
    if not info_dict:
        return redirect('/index/')

    return render(request, 'home.html', {'info_dict': info_dict})


def user_exit(request):
    logout(request)
    return redirect('/home/')


def upload(request):
    if request.method == 'POST' and request.FILES['image']:
        # 获取用户信息
        info_dict = request.session.get('info')
        user_id = info_dict['id']

        # 获取上传图片，并保存
        input_image = request.FILES['image']
        output_image = 'output/' + input_image.name
        models.Image.objects.create(user_id=user_id, input_img=input_image, output_img=output_image)

        # 加载模型，实现风格转化
        model = torch.hub.load("static/animegan2/", "generator", pretrained="face_paint_512_v2", source='local').eval()
        # 加载图片生成函数
        face2paint = torch.hub.load("static/animegan2/", "face2paint", size=512, source='local')

        # 获取要转换的图片，并在output/目录下生成转换后的图片auth_user_user_permissions
        img = Image.open('static/media/input/' + input_image.name).convert("RGB")
        output_path = 'static/media/output/'
        out_img = face2paint(model, img)
        out_img.save(os.path.join(output_path, f'{input_image.name}'))

        return redirect('/home/')


def img_list(request):
    info_dict = request.session.get('info')
    if not info_dict:
        return redirect('/index/')

    info_dict = request.session.get('info')
    user_id = info_dict['id']
    queryset = models.Image.objects.filter(user_id=user_id)

    return render(request, 'img_list.html', {'queryset': queryset, 'info_dict': info_dict})


def img_del(request):
    info_dict = request.session.get('info')
    if not info_dict:
        return redirect('/index/')

    # 根据id获取要删除的图片
    del_id = request.GET.get('id')
    del_img = models.Image.objects.get(id=del_id)

    # 删除服务器图片
    del_input_img_path = del_img.input_img.path
    del_output_img_path = del_img.output_img.path

    try:
        os.remove(del_input_img_path)
    except OSError as e:
        print(f'Error deleting image: {str(e)}')

    try:
        os.remove(del_output_img_path)
    except OSError as e:
        print(f'Error deleting image: {str(e)}')

    models.Image.objects.filter(id=del_id).delete()

    return redirect('/home/list/')


def img_view(request):
    # 根据id获取要浏览的图片
    img_id = request.GET.get('id')
    down_img = models.Image.objects.get(id=img_id)
    down_image = down_img.output_img
    print(down_image)

    file_path = os.path.join(settings.MEDIA_ROOT, str(down_image))
    if os.path.exists(file_path):
        # 使用 Django 的 FileResponse 将图片作为文件流返回给前端
        return FileResponse(open(file_path, 'rb'))

    # 如果指定图片不存在，则返回一个 404 错误响应
    raise Http404('未找到该文件')


def img_down(request):
    # 根据id获取要下载的图片
    img_id = request.GET.get('id')
    down_img = models.Image.objects.get(id=img_id)
    down_image = down_img.output_img
    print(down_image)

    file_path = os.path.join(settings.MEDIA_ROOT, str(down_image))
    if os.path.exists(file_path):
        # 使用 Django 的 FileResponse 将图片作为文件流返回给前端
        response = FileResponse(open(file_path, 'rb'), as_attachment=True)
        response['Content-Disposition'] = f'attachment; filename="{down_image.name}"'
        return response

    # 如果指定图片不存在，则返回一个 404 错误响应
    raise Http404('未找到该文件')


def user_list(request):
    info_dict = request.session.get('info')
    if not info_dict:
        return redirect('/manager/login')

    u_list = models.User.objects.all()
    return render(request, 'user_list.html', {'all_user': u_list, 'info_dict': info_dict})


def user_del(request):
    info_dict = request.session.get('info')
    if not info_dict:
        return redirect('/user/list/')

    # 根据id获取要删除的用户
    del_id = request.GET.get('id')

    # 获取要删除用户的所有图片
    user_img = models.Image.objects.filter(user_id=del_id)

    # 判断用户是否有图片
    if user_img:

        # 如果有，则先删除图片数据
        imglist = models.Image.objects.filter(user_id=del_id)
        # 由于用户可能有多个图片，主要是要同步删除服务器下的图片，所以用for循环遍历，依次删除
        for img in imglist:
            # 获取图片信息
            img_id = img.id
            del_input_img_path = img.input_img.path
            del_output_img_path = img.output_img.path

            # 删除服务器图片
            try:
                os.remove(del_input_img_path)
            except OSError as e:
                print(f'Error deleting image: {str(e)}')

            try:
                os.remove(del_output_img_path)
            except OSError as e:
                print(f'Error deleting image: {str(e)}')

            # 删除数据库图片数据
            models.Image.objects.filter(id=img_id).delete()

        # 再删除用户
        models.User.objects.filter(id=del_id).delete()
    else:
        # 如果用户没有图片，则直接删除用户
        models.User.objects.filter(id=del_id).delete()

    return redirect('/user/list/')
