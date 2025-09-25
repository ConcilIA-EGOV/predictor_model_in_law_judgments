import numpy as np
from util.parameters import FAIXAS_EXTRAVIO, FAIXAS_ATRASO, FAIXAS_DANO, log_file

def generate_range(value, interval_values=[]) -> int:
    for i, interval in enumerate(interval_values):
        if value < interval:
            return i
    return len(interval_values)


def hour_to_float(value, interval_values=[]):
    splits = value.split(":")
    last = len(splits) - 1
    minutes = float(splits[-last].strip()) / 60
    hours = float(splits[-(last + 1)].strip())
    f_value = hours + minutes
    f_value = generate_range(f_value, interval_values)
    return f_value


def format_comma_strings(value:str, interval_values=[]):
    if ',' in value:
        f_value = float(value.replace(',', '.'))
    else:
        f_value = float(value)
    f_value = generate_range(f_value, interval_values)
    return f_value


def format_binario(value, anomaly=0, yes=1, no=0) -> int:
    if type(value) == float:
        if np.isnan(value):
            return anomaly
    if type(value) == int:
        if value == 1:
            return yes
        if value == 0:
            return no
        if value == -1:
            return anomaly
    if type(value) == bool:
        if value:
            return yes
        else:
            return no
    if value in ['S', 's', 'Y', 'y', 'Sim', 'sim', 'SIM', 'YES', 'Yes', 'yes', '1']:
        return yes
    if value in ['N', 'n', 'Não', 'não', 'NÃO', 'NO', 'No', 'no', '0']:
        return no
    log_file.write(f" -> Valor binário não reconhecido: {value}\n")
    return anomaly
    

def format_intervalo(value, interval_values=[]):
    if type(value) == int:
        return value
    if type(value) == float:
        if np.isnan(value):
            return 0
        return generate_range(value, interval_values)
    if ',' in value:
        return format_comma_strings(value, interval_values)
    if ':' in value:
        return hour_to_float(value, interval_values)
    else:
        log_file.write(f" -> Valor de intervalo não reconhecido: {value}\n")
        return value


FUNCTIONS = {
    'sentenca': lambda x: x,
    'direito_de_arrependimento': lambda x: format_binario(x),
    'descumprimento_de_oferta': lambda x: format_binario(x),
    'extravio_definitivo': lambda x: format_binario(x),
    'intervalo_extravio_temporario': lambda x: format_intervalo(x, FAIXAS_EXTRAVIO),
    'faixa_intervalo_extravio_temporario': lambda x: int(x),
    'violacao_furto_avaria': lambda x: format_binario(x),
    'cancelamento/alteracao_destino': lambda x: format_binario(x),
    'cancelamento': lambda x: format_binario(x),
    'intervalo_atraso': lambda x: format_intervalo(x, FAIXAS_ATRASO),
    'faixa_intervalo_atraso': lambda x: int(x),
    'culpa_exclusiva_consumidor': lambda x: format_binario(x),
    'condicoes_climaticas/fechamento_aeroporto': lambda x: format_binario(x),
    'fechamento_aeroporto': lambda x: format_binario(x),
    'noshow': lambda x: format_binario(x),
    'overbooking': lambda x: format_binario(x),
    'assistencia_cia_aerea': lambda x: format_binario(x, -1),
    'hipervulneravel': lambda x: format_binario(x),
    'dano_moral_individual': lambda x: format_comma_strings(x, FAIXAS_DANO),
    'faixa_dano_moral_individual': lambda x: int(x),
    'Dano-Moral': lambda x: int(x),
}
