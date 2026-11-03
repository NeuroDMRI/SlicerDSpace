"""Validate resource contracts before work, following strict workflow-node admission."""

import math


def validate_request(request, limits, *, check_budget=True):
    for field in ('cpu','gpu','io'):
        value=request.get(field,0)
        if type(value) is not int or value < (1 if field=='cpu' else 0):
            raise ValueError('Stage '+field+' must be a positive CPU count or nonnegative slot count')
    memory=request.get('memory_gb',0)
    if isinstance(memory,bool) or not isinstance(memory,(int,float)) or not math.isfinite(memory) or memory<=0:
        raise ValueError('Stage memory must be finite and positive')
    duration=request.get('duration_seconds',0)
    if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<0:
        raise ValueError('Stage duration estimate must be finite and nonnegative')
    if check_budget and any(request.get(field,0)>limits[field] for field in ('cpu','memory_gb','gpu','io')):
        raise ValueError('Stage request exceeds the cohort resource budget')


def stage_request(config, subject, name):
    request=dict(cpu=config['stage_cpus'],memory_gb=config['stage_memory_gb'],gpu=0,io=0,duration_seconds=0)
    request.update(config.get('subject_profiles',{}).get(subject,{}).get(name,{}))
    request['memory_floor_gb']=request['memory_gb']
    key=request.get('profile_key')
    if config.get('history_path') and key:
        from shared.cohort_profile import read_history, history_estimate
        feedback=history_estimate(read_history(config['history_path']),key)
        if feedback:
            if not (config.get('planning') or {}).get('memory_reservations_fixed'):
                request['memory_gb']=max(request['memory_gb'],math.ceil(feedback['memory_gb']*4)/4)
            request['duration_seconds']=feedback['duration_seconds']
            request['history_samples']=feedback['samples']
            request['estimate_source']='header + completed cohort measurements'
    return request
