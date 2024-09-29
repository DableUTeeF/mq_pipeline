from hat.models import hat_model  # need to be imported to register
import pika
import os
import json
from basicsr.models import build_model
from basicsr.utils import imfrombytes, img2tensor, tensor2img
import time
import torch
import cv2
from collections import OrderedDict
import io


_ = hat_model  # just to stop pycharm from complaining
checkpoint = 'cp/Real_HAT_GAN_SRx4.pth'
opt = {'name': 'HAT_GAN_Real_SRx4', 'model_type': 'HATModel', 'scale': 4, 'num_gpu': 1, 'manual_seed': 0, 'tile': OrderedDict([('tile_size', 512), ('tile_pad', 32)]), 'datasets': OrderedDict([('test_1', OrderedDict([('name', 'custom'), ('type', 'SingleImageDataset'), ('dataroot_lq', 'input_dir'), ('io_backend', OrderedDict([('type', 'disk')])), ('phase', 'test'), ('scale', 4)]))]), 'network_g': OrderedDict([('type', 'HAT'), ('upscale', 4), ('in_chans', 3), ('img_size', 64), ('window_size', 16), ('compress_ratio', 3), ('squeeze_factor', 30), ('conv_scale', 0.01), ('overlap_ratio', 0.5), ('img_range', 1.0), ('depths', [6, 6, 6, 6, 6, 6]), ('embed_dim', 180), ('num_heads', [6, 6, 6, 6, 6, 6]), ('mlp_ratio', 2), ('upsampler', 'pixelshuffle'), ('resi_connection', '1conv')]), 'path': OrderedDict([('pretrain_network_g', checkpoint), ('strict_load_g', True), ('param_key_g', 'params_ema'), ('results_root', '/home/palm/PycharmProjects/hat/options/test/HAT_GAN_Real_SRx4.yml/results/HAT_GAN_Real_SRx4'), ('log', '/home/palm/PycharmProjects/hat/options/test/HAT_GAN_Real_SRx4.yml/results/HAT_GAN_Real_SRx4'), ('visualization', '/home/palm/PycharmProjects/hat/options/test/HAT_GAN_Real_SRx4.yml/results/HAT_GAN_Real_SRx4/visualization')]), 'val': OrderedDict([('save_img', True), ('suffix', None)]), 'dist': False, 'rank': 0, 'world_size': 1, 'auto_resume': False, 'is_train': False}
# opt = OrderedDict([('name', 'HAT-S_SRx4'), ('model_type', 'HATModel'), ('scale', 4), ('num_gpu', 1), ('manual_seed', 0), ('datasets', OrderedDict([('test_1', OrderedDict([('name', 'Set5'), ('type', 'PairedImageDataset'), ('dataroot_gt', './datasets/Set5/GTmod4'), ('dataroot_lq', './datasets/Set5/LRbicx4'), ('io_backend', OrderedDict([('type', 'disk')])), ('phase', 'test'), ('scale', 4)])), ('test_2', OrderedDict([('name', 'Set14'), ('type', 'PairedImageDataset'), ('dataroot_gt', './datasets/Set14/GTmod4'), ('dataroot_lq', './datasets/Set14/LRbicx4'), ('io_backend', OrderedDict([('type', 'disk')])), ('phase', 'test'), ('scale', 4)])), ('test_3', OrderedDict([('name', 'Urban100'), ('type', 'PairedImageDataset'), ('dataroot_gt', './datasets/urban100/GTmod4'), ('dataroot_lq', './datasets/urban100/LRbicx4'), ('io_backend', OrderedDict([('type', 'disk')])), ('phase', 'test'), ('scale', 4)])), ('test_4', OrderedDict([('name', 'BSDS100'), ('type', 'PairedImageDataset'), ('dataroot_gt', './datasets/BSDS100/GTmod4'), ('dataroot_lq', './datasets/BSDS100/LRbicx4'), ('io_backend', OrderedDict([('type', 'disk')])), ('phase', 'test'), ('scale', 4)])), ('test_5', OrderedDict([('name', 'Manga109'), ('type', 'PairedImageDataset'), ('dataroot_gt', './datasets/manga109/GTmod4'), ('dataroot_lq', './datasets/manga109/LRbicx4'), ('io_backend', OrderedDict([('type', 'disk')])), ('phase', 'test'), ('scale', 4)]))])), ('network_g', OrderedDict([('type', 'HAT'), ('upscale', 4), ('in_chans', 3), ('img_size', 64), ('window_size', 16), ('compress_ratio', 24), ('squeeze_factor', 24), ('conv_scale', 0.01), ('overlap_ratio', 0.5), ('img_range', 1.0), ('depths', [6, 6, 6, 6, 6, 6]), ('embed_dim', 144), ('num_heads', [6, 6, 6, 6, 6, 6]), ('mlp_ratio', 2), ('upsampler', 'pixelshuffle'), ('resi_connection', '1conv')])), ('path', OrderedDict([('pretrain_network_g', 'cp/HAT-S_SRx4.pth'), ('strict_load_g', True), ('param_key_g', 'params_ema'), ('results_root', '/home/palm/PycharmProjects/hat/options/test/HAT-S_SRx4.yml/results/HAT-S_SRx4'), ('log', '/home/palm/PycharmProjects/hat/options/test/HAT-S_SRx4.yml/results/HAT-S_SRx4'), ('visualization', '/home/palm/PycharmProjects/hat/options/test/HAT-S_SRx4.yml/results/HAT-S_SRx4/visualization')])), ('val', OrderedDict([('save_img', True), ('suffix', None), ('metrics', OrderedDict([('psnr', OrderedDict([('type', 'calculate_psnr'), ('crop_border', 4), ('test_y_channel', True)])), ('ssim', OrderedDict([('type', 'calculate_ssim'), ('crop_border', 4), ('test_y_channel', True)]))]))])), ('dist', False), ('rank', 0), ('world_size', 1), ('auto_resume', False), ('is_train', False)])
model = build_model(opt)
model.net_g.eval()


@torch.no_grad()
def process(file_bytes):
    try:
        img_lq = imfrombytes(file_bytes, float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True).cuda()
        if img_lq.size(1) * img_lq.size(2) > 800*600:
            return {'error': 'Image is too big. Only supports up to 800x600 pixels', 'success': False}
        width_pad = None
        height_pad = None
        if img_lq.size(1) % 16 != 0:
            width_pad = ((img_lq.size(1) // 16 + 1) * 16) - img_lq.size(1)
            pads = img_lq[:, -1:, :].expand(img_lq.size(0), width_pad, img_lq.size(2))
            img_lq = torch.cat([img_lq, pads.cuda()], dim=1)
        if img_lq.size(2) % 16 != 0:
            height_pad = ((img_lq.size(2) // 16 + 1) * 16) - img_lq.size(2)
            pads = img_lq[:, :, -1:].expand(img_lq.size(0), img_lq.size(1), height_pad)
            img_lq = torch.cat([img_lq, pads.cuda()], dim=2)
        model.lq = img_lq[None]
        model.pre_process()
        with torch.no_grad():
            model.process()
        model.post_process()
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        if width_pad is not None:
            sr_img = sr_img[:-width_pad*4]
        if height_pad is not None:
            sr_img = sr_img[:, :-height_pad*4]
        res, im_png = cv2.imencode(".png", sr_img)
        return im_png.tobytes()
    except Exception as e:
        return {'error': str(e), 'success': False}


def on_request(ch, method, props, body):
    response = process(body)
    if isinstance(response, dict):
        response = json.dumps(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=response)
    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    url = os.environ.get('CLOUDAMQP_URL', 'amqp://guest:guest@sr-mq/%2f')  # Taz check!
    time.sleep(5)
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)

    channel = connection.channel()
    channel.queue_declare(queue='capgen_queue')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='capgen_queue', on_message_callback=on_request)

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
