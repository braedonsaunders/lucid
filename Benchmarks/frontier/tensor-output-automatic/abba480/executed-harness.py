#!/usr/bin/env python3
"""Own an isolated Chrome/MessageChannel/native comparison and clean up every process."""
import argparse
import gzip
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
    ap.add_argument('--extension',type=Path,
                    help='Install this unpacked companion through Chrome CDP in the owned browser profile')
    ap.add_argument('--samples',type=int,default=30)
    ap.add_argument('--candidate-model-stem',default='direct2x_trained_')
    ap.add_argument('--candidate-sharpness',type=float,default=.4)
    ap.add_argument('--candidate-tensor-output',action='store_true',
                    help='Enable verified FP32 packing only in the owned ephemeral candidate process')
    ap.add_argument('--candidate-automatic-tensor',action='store_true',
                    help='Exercise bundled runtime admission without a model override or experimental opt-in')
    ap.add_argument('--order',nargs='+',choices=['shipping','candidate'],default=['shipping','candidate','candidate','shipping'])
    ap.add_argument('--presentation-trace',action='store_true',help='Record actual draw acknowledgments in the extension iframe')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.candidate_automatic_tensor and args.candidate_tensor_output:
        ap.error('automatic admission and explicit tensor experiments are separate modes')
    if not 0 <= args.candidate_sharpness <= 2 or not args.candidate_model_stem.replace('_','').isalnum():
        ap.error('bounded sharpness and an alphanumeric/underscore model stem required')
    if args.out.exists():ap.error('fresh browser evidence directory required')
    if args.extension and not (args.extension/'manifest.json').is_file():
        ap.error('extension manifest required')
    if args.extension:
        ignored=json.loads(args.config.read_text()).get('browser',{}).get('launchOptions',{}).get('ignoreDefaultArgs',[])
        if not isinstance(ignored,list) or '--disable-extensions' not in ignored:
            ap.error('installed-extension config must ignore the Playwright --disable-extensions default')
    if not 1 <= args.samples <= 3600 or (args.presentation_trace and not args.extension):
        ap.error('1..3600 samples required; presentation tracing requires an installed extension')
    for port in [48111,48112,48113]:
        with socket.socket() as check:check.bind(('127.0.0.1',port))
    args.out.mkdir(parents=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('LUCID_')}
    env['PLAYWRIGHT_SKIP_BROWSER_GC']='1'
    server_log=(args.out/'server.log').open('w')
    server=subprocess.Popen(['python3','-m','http.server','48113','--bind','127.0.0.1','--directory',str(args.fixture.resolve())],stdout=server_log,stderr=subprocess.STDOUT)
    app=None;session=None;app_log=None
    report={'purpose':f'{"headed" if args.headed else "headless"} Chrome native playback comparison; source and delivered dimensions recorded in run telemetry',
        'clip_sha256':digest(args.fixture/'video.mp4'),'app_sha256':digest(args.app),
        'script_sha256':digest(__file__),'browser_config_sha256':digest(args.config),
        'scope':'Real companion scripts and MessageChannels, isolated native app/ports; not installed-extension or third-party CSP coverage',
        'order':args.order,'samples_per_run':args.samples,'warmup_seconds':5,
        'candidate_model_stem':args.candidate_model_stem,'candidate_sharpness':args.candidate_sharpness,
        'candidate_tensor_output':args.candidate_tensor_output,
        'candidate_automatic_tensor':args.candidate_automatic_tensor,
        'presentation_trace':args.presentation_trace,'runs':[],'complete':False}
    probe_path=Path(__file__).with_name('browser_draw_probe.js')
    if args.presentation_trace:report['presentation_probe_sha256']=digest(probe_path)
    report['fixture_files']={p.name:digest(p) for p in args.fixture.iterdir() if p.suffix in ('.js','.html','.json')}
    if args.extension:
        report['scope']='Installed unpacked companion in an owned Chrome profile; real extension messaging and extension iframe; local fixture, not third-party CSP coverage'
        report['extension_files']={str(p.relative_to(args.extension)):digest(p) for p in args.extension.rglob('*') if p.is_file()}
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
                'LUCID_COMPUTE_UNITS':'gpu','LUCID_MODEL_STEM':args.candidate_model_stem if label=='candidate' else 'SPAN_x4_ch32utc_'}
            if label=='candidate' and args.candidate_tensor_output:
                native_env['LUCID_EXPERIMENTAL_TENSOR_OUTPUT']='1'
            if label=='candidate' and args.candidate_automatic_tensor:
                native_env.pop('LUCID_MODEL_STEM')
            app=subprocess.Popen([str(args.app.resolve()),'-strength','standard'],env=native_env,stdout=app_log,stderr=subprocess.STDOUT)
            cli('open','about:blank' if args.extension else 'http://127.0.0.1:48113','--browser','chrome','--config',str(args.config.resolve()),*(['--headed'] if args.headed else []))
            installation=None
            if args.presentation_trace:
                js('''async page => {
                  await page.context().addInitScript({content: DRAW_PROBE});
                  return {purpose:'presentation-probe-installed'};
                }'''.replace('DRAW_PROBE',json.dumps(probe_path.read_text())))
            if args.extension:
                installation=js('''async page => {
                  const cdp=await page.context().browser().newBrowserCDPSession();
                  try {
                    const installed=await cdp.send('Extensions.loadUnpacked',{path:EXTENSION_PATH,enableInIncognito:true});
                    const inventory=await cdp.send('Extensions.getExtensions');
                    await page.goto('http://127.0.0.1:48113');
                    return {purpose:'extension-install',installed,inventory};
                  } finally { await cdp.detach(); }
                }'''.replace('EXTENSION_PATH',json.dumps(str(args.extension.resolve()))))
            cli('snapshot')
            setup=js('''async page => {
              await page.waitForFunction(() => window.controlSocket?.readyState===1 && window.latestStatus, null, {timeout:20000});
              const enabledAt = Date.now();
              await page.evaluate(() => { window.controlSocket.send(JSON.stringify({type:'control',enabled:true,tuning:{sharpness:GAIN}})); document.querySelector('video').play(); });
              await page.waitForFunction(() => window.latestStatus?.enhancing && window.latestStatus?.presentedFPS>0, null, {timeout:30000});
              const readiness = Date.now() - enabledAt;
              await page.waitForTimeout(5000);
              return {...await page.evaluate(() => ({purpose:'setup',status:window.latestStatus,browser:navigator.userAgent,video:{w:document.querySelector('video').videoWidth,h:document.querySelector('video').videoHeight},viewport:[innerWidth,innerHeight,devicePixelRatio]})), enable_to_positive_presented_fps_ms:readiness};
            }'''.replace('GAIN',str(args.candidate_sharpness) if label=='candidate' else '.75'))
            if app.poll() is not None:raise RuntimeError('native process ended during setup')
            chunks=[]
            for start in range(0,args.samples,30):
                chunk=js('''async page => {
              const result=await page.evaluate(async () => {
                const startedEpoch=performance.timeOrigin+performance.now();
                const samples=[];
                for(let i=0;i<SAMPLE_COUNT;i++) { await new Promise(resolve=>setTimeout(resolve,1000)); samples.push({time:performance.now(),status:window.latestStatus,frameStats:document.documentElement.dataset.lucidFrames,gateStats:document.documentElement.dataset.lucidGate}); }
                return {purpose:'cadence',startedEpoch,endedEpoch:performance.timeOrigin+performance.now(),samples,video:document.querySelector('video').getVideoPlaybackQuality().toJSON?.() ?? {totalVideoFrames:document.querySelector('video').getVideoPlaybackQuality().totalVideoFrames,droppedVideoFrames:document.querySelector('video').getVideoPlaybackQuality().droppedVideoFrames}};
              });
              result.frames=[];
              for(const frame of page.frames()) result.frames.push({url:frame.url(),canvases:await frame.locator('canvas').evaluateAll(nodes=>nodes.map(n=>({width:n.width,height:n.height,alpha:getComputedStyle(n).opacity})))});
              return result;
            }'''.replace('SAMPLE_COUNT',str(min(30,args.samples-start))))
                chunk['native_rss_kib']=int(subprocess.check_output(['ps','-o','rss=','-p',str(app.pid)],text=True).strip())
                chunks.append(chunk)
                (args.out/f'{index}-{label}-chunks.json').write_text(json.dumps(chunks,indent=2)+'\n')
                print(index,label,start+len(chunk['samples']),'samples collected',flush=True)
            samples={**chunks[-1],'startedEpoch':chunks[0]['startedEpoch'],
                     'samples':[s for chunk in chunks for s in chunk['samples']],
                     'native_memory_samples':[{'at':chunk['endedEpoch'],'rss_kib':chunk['native_rss_kib']} for chunk in chunks]}
            if args.presentation_trace:
                trace=js('''async page => {
                  const surfaces=[];
                  for(const frame of page.frames()) {
                    const samples=await frame.evaluate(()=>globalThis.__lucidDrawProbe??[]);
                    if(samples.length) surfaces.push({url:frame.url(),samples});
                  }
                  return {purpose:'presentation-trace',surfaces};
                }''')
                if not trace['surfaces']:raise RuntimeError('no actual surface presentation acknowledgments recorded')
                path=args.out/f'{index}-{label}-presentations.json.gz'
                path.write_bytes(gzip.compress(json.dumps(trace).encode(),mtime=0))
                samples['presentation_trace']={'file':path.name,'sha256':digest(path)}
            if not all(x['status']['enhancing'] and x['status']['presentedFPS']>0 for x in samples['samples']):
                raise RuntimeError('playback stalled or enhancement stopped during sample')
            off=js('''async page => {
              await page.evaluate(()=>window.controlSocket.send(JSON.stringify({type:'control',enabled:false})));
              await page.waitForTimeout(2000);
              return await page.evaluate(()=>({purpose:'off',enabled:window.latestStatus.enabled,enhancing:window.latestStatus.enhancing}));
            }''')
            if off['enabled'] or off['enhancing']:raise RuntimeError('Off did not settle')
            report['runs'].append({'variant':label,'installation':installation,'setup':setup,'measurement':samples,'off':off});save()
            print(index,label,args.samples,'samples complete',flush=True)
            cli('close');session=None
            app.terminate();app.wait(timeout=15);app=None;app_log.close();app_log=None
        report['complete']=True;save()
    except Exception as error:
        report['failure']=repr(error)
        report['native_exit_code']=app.poll() if app else None
        if session:
            try:
                report['failure_browser_state']=js('''async page => {
                  const state=await page.evaluate(()=>({status:window.latestStatus,statuses:window.statuses,
                    socket:window.controlSocket?.readyState,frames:document.documentElement.dataset.lucidFrames,
                    gate:document.documentElement.dataset.lucidGate,
                    videos:[...document.querySelectorAll('video')].map(v=>({width:v.videoWidth,height:v.videoHeight,
                      paused:v.paused,time:v.currentTime,readyState:v.readyState,error:v.error?.message}))}));
                  return {purpose:'failure-diagnostic',state,frames:page.frames().map(f=>f.url()),
                    workers:page.context().serviceWorkers().map(w=>w.url()),browser:await page.evaluate(()=>navigator.userAgent)};
                }''')
            except Exception as diagnostic_error:
                report['failure_diagnostic_error']=repr(diagnostic_error)
        save()
        raise
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
