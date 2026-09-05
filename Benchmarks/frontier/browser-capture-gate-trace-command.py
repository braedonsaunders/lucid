#!/usr/bin/env python3
"""Own an isolated Chrome/MessageChannel/native comparison and clean up every process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import socket
import time
import urllib.request


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--headed',action='store_true')
    ap.add_argument('--app',type=Path,required=True)
    ap.add_argument('--fixture',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('fresh browser evidence directory required')
    for port in [48111,48112,48113]:
        with socket.socket() as check:check.bind(('127.0.0.1',port))
    args.out.mkdir(parents=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('LUCID_')}
    env['PLAYWRIGHT_SKIP_BROWSER_GC']='1'
    server_log=(args.out/'server.log').open('w')
    server=subprocess.Popen(['python3','-m','http.server','48113','--bind','127.0.0.1','--directory',str(args.fixture.resolve())],stdout=server_log,stderr=subprocess.STDOUT)
    app=None;session=None;app_log=None
    report={'purpose':f'ABBA {"headed" if args.headed else "headless"} Chrome native playback comparison at 640x360 to 1280x720',
        'clip_sha256':digest(args.fixture/'video.mp4'),'app_sha256':digest(args.app),
        'script_sha256':digest(__file__),'browser_config_sha256':digest(args.config),
        'scope':'Real companion scripts and MessageChannels, isolated native app/ports; not installed-extension or third-party CSP coverage',
        'order':['candidate'],'runs':[],'complete':False}
    def save(): (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def cli(*arguments,timeout=55):
        result=subprocess.run(['playwright-cli',f'-s={session}',*arguments],env=env,capture_output=True,text=True,timeout=timeout)
        with (args.out/'cli.log').open('a') as log: log.write(result.stdout+result.stderr)
        result.check_returncode();return result.stdout
    def js(code):
        path=args.out/'command.js';path.write_text(code)
        output=cli('--raw','run-code','--filename',str(path.resolve()))
        values=[json.loads(line) for line in output.splitlines() if line.startswith('{"purpose"')]
        if len(values)!=1:raise RuntimeError('no unique browser telemetry result: '+output[-1200:])
        return values[0]
    try:
        for attempt in range(30):
            if server.poll() is not None:raise RuntimeError('owned fixture server failed')
            try:
                urllib.request.urlopen('http://127.0.0.1:48113',timeout=1).close();break
            except OSError:time.sleep(.1)
        for index,label in enumerate(report['order']):
            session=f'lucid-cadence-{os.getpid()}-{index}'
            app_log=(args.out/f'{index}-{label}-app.log').open('w')
            native_env={**env,'LUCID_EPHEMERAL':'1','LUCID_BRIDGE_PORT':'48111','LUCID_TOKEN_PORT':'48112',
                'LUCID_COMPUTE_UNITS':'gpu','LUCID_MODEL_STEM':'direct2x_trained_' if label=='candidate' else 'SPAN_x4_ch32utc_'}
            app=subprocess.Popen([str(args.app.resolve()),'-strength','standard'],env=native_env,stdout=app_log,stderr=subprocess.STDOUT)
            cli('open','http://127.0.0.1:48113','--browser','chrome','--config',str(args.config.resolve()),*(['--headed'] if args.headed else []))
            cli('snapshot')
            setup=js('''async page => {
              await page.waitForFunction(() => window.controlSocket?.readyState===1 && window.latestStatus, null, {timeout:20000});
              await page.evaluate(() => { window.controlSocket.send(JSON.stringify({type:'control',enabled:true,tuning:{sharpness:GAIN}})); document.querySelector('video').play(); });
              await page.waitForFunction(() => window.latestStatus?.enhancing && window.latestStatus?.presentedFPS>0, null, {timeout:30000});
              await page.waitForTimeout(5000);
              return await page.evaluate(() => ({purpose:'setup',status:window.latestStatus,browser:navigator.userAgent,video:{w:document.querySelector('video').videoWidth,h:document.querySelector('video').videoHeight},viewport:[innerWidth,innerHeight,devicePixelRatio]}));
            }'''.replace('GAIN','.4' if label=='candidate' else '.75'))
            if app.poll() is not None:raise RuntimeError('native process ended during setup')
            samples=js('''async page => {
              const result=await page.evaluate(async () => {
                const samples=[];
                for(let i=0;i<10;i++) { await new Promise(resolve=>setTimeout(resolve,1000)); samples.push({time:performance.now(),status:window.latestStatus,frameStats:document.documentElement.dataset.lucidFrames}); }
                return {purpose:'cadence',samples,gateTrace:window.gateTrace,video:document.querySelector('video').getVideoPlaybackQuality().toJSON?.() ?? {totalVideoFrames:document.querySelector('video').getVideoPlaybackQuality().totalVideoFrames,droppedVideoFrames:document.querySelector('video').getVideoPlaybackQuality().droppedVideoFrames}};
              });
              result.frames=[];
              for(const frame of page.frames()) result.frames.push({url:frame.url(),canvases:await frame.locator('canvas').evaluateAll(nodes=>nodes.map(n=>({width:n.width,height:n.height,alpha:getComputedStyle(n).opacity})))});
              return result;
            }''')
            if not all(x['status']['enhancing'] and x['status']['presentedFPS']>0 for x in samples['samples']):
                raise RuntimeError('playback stalled or enhancement stopped during sample')
            off=js('''async page => {
              await page.evaluate(()=>window.controlSocket.send(JSON.stringify({type:'control',enabled:false})));
              await page.waitForTimeout(2000);
              return await page.evaluate(()=>({purpose:'off',enabled:window.latestStatus.enabled,enhancing:window.latestStatus.enhancing}));
            }''')
            if off['enabled'] or off['enhancing']:raise RuntimeError('Off did not settle')
            report['runs'].append({'variant':label,'setup':setup,'measurement':samples,'off':off});save()
            print(index,label,'10 samples complete',flush=True)
            cli('close');session=None
            app.terminate();app.wait(timeout=15);app=None;app_log.close();app_log=None
        report['complete']=True;save()
    finally:
        if session:
            try:
                subprocess.run(['playwright-cli',f'-s={session}','close'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20,check=True)
            except subprocess.SubprocessError:
                print('Session cleanup needs inspection:',session,flush=True)
        if app and app.poll() is None:
            app.terminate()
            try:app.wait(timeout=10)
            except subprocess.TimeoutExpired:app.kill();app.wait()
        if app_log:app_log.close()
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        server_log.close()


if __name__=='__main__':main()
