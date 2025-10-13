import numpy as np
from util.parameters import FAIXAS_EXTRAVIO, FAIXAS_ATRASO, TARGET
from util.parameters import  log_file_preprocessing, append_to_data_log_list
# log file to record changes
log_file = log_file_preprocessing
# Global variable to keep track of the current column being processed
current_column = ""
# Global variable to indicate if it's the first run
first_run = False


def generate_range(value:int|float, interval_values=[]) -> int:
    for i, interval in enumerate(interval_values):
        if value < interval:
            return i
    return len(interval_values)


def hour_to_int(value:str, interval_values=[]) -> int:
    """
    Converte uma string de hora no formato HH:MM para um valor inteiro representando a hora.
    """
    ttp = 'inteiro'
    splits = value.split(":")
    last = len(splits) - 1
    minutes = float(splits[-last].strip()) / 60
    hours = float(splits[-(last + 1)].strip())
    f_value = int(hours + minutes)
    if interval_values:
        ttp = 'faixa'
        f_value = generate_range(f_value, interval_values)
    global first_run
    if first_run:
        msg = f"Coluna '{current_column}': convertendo hora no formato '{value}' para {ttp} como '{f_value}'"
        log_file.write(f" -> {msg}\n")
        append_to_data_log_list("Alteracoes nas Features", msg)
        first_run = False
    return f_value


def format_money(value) -> int:
    out_value = 0
    original_value = value
    value = str(value).strip()
    if 'R$' in value:
        value = value.replace('R$', '').strip()
    if ',' in value:
        # se forem as vírgulas dos centos,
        # simplesmente remove os centos
        tmp = value.split(',')
        is_cent = (len(tmp[-1]) == 2)
        if value.count(',') == 1 and is_cent:
            value = value[:-3]
        # se forem vírgulas de milhares, remove-as
        elif not is_cent:
            value = value.replace(',', '')
        else:
            value = value.replace(',', '.')
    out_value = int(float(value))
    global first_run
    if first_run:
        msg = f"Coluna '{current_column}': convertendo strings monetarias como '{original_value}' para int como '{out_value}'"
        log_file.write(f" -> {msg}\n")
        append_to_data_log_list("Alteracoes nas Features", msg)
        first_run = False
    return out_value


def format_binario(value, anomaly: int=0, yes: int=1, no: int=0) -> int:
    output = 0
    if type(value) == float:
        if np.isnan(value):
            output = anomaly
    elif type(value) == int:
        if value == 1:
            output = yes
        if value == 0:
            output = no
        if value == -1:
            output = anomaly
    elif type(value) == bool:
        if value:
            output = yes
        else:
            output = no
    elif type(value) == str:
        if value in ['S', 's', 'Y', 'y', 'Sim', 'sim', 'SIM', 'YES', 'Yes', 'yes', '1']:
            output = yes
        if value in ['N', 'n', 'Não', 'não', 'NÃO', 'NO', 'No', 'no', '0']:
            output = no
        if value in ['-1', '-']:
            output = anomaly
    else:
        log_file.write(f" -> Valor binario não reconhecido: {value}\n")
        output = anomaly
    global first_run
    if first_run:
        msg = f"Coluna '{current_column
            }': convertendo valores binarios como '{value
            }' para '{yes}, {no}, ou {anomaly}'"
        log_file.write(f" -> {msg}\n")
        append_to_data_log_list("Alteracoes nas Features", msg)
        first_run = False
    return output
    

def format_intervalo(value, interval_values=[]) -> int:
    output = 0
    if type(value) == int:
        output =  value
    elif type(value) == float and not np.isnan(value):
        output = int(value)
    elif type(value) == str:
        f_out = 0
        if ',' in value:
            f_out = format_money(value)
        elif ':' in value:
            f_out =  hour_to_int(value, interval_values)
        output = generate_range(f_out, interval_values)
        if type(output) != int:
            output = int(output)
    else:
        log_file.write(f" -> Valor de intervalo não reconhecido: {value}\n")
    global first_run
    if first_run:
        msg = f"Coluna '{current_column
            }': convertendo valores {type(value).__name__} como '{value
            }' para faixa como '{output
            }' com valores de intervalo {interval_values}"
        log_file.write(f" -> {msg}\n")
        append_to_data_log_list("Alteracoes nas Features", msg)
        first_run = False
    return output


FUNCTIONS = {
    TARGET: lambda x: format_money(x),
    
    'ano': lambda x: int(x),
    'sentenca': lambda x: int(x),
    'semestre': lambda x: int(x),
    'trimestre': lambda x: int(x),
    
    'noshow': lambda x: format_binario(x),
    'overbooking': lambda x: format_binario(x),
    'cancelamento': lambda x: format_binario(x),
    'hipervulneravel': lambda x: format_binario(x),
    'extravio_definitivo': lambda x: format_binario(x),
    'desamparo': lambda x: format_binario(x, anomaly=-1),
    'violacao_furto_avaria': lambda x: format_binario(x),
    'descumprimento_de_oferta': lambda x: format_binario(x),
    'direito_de_arrependimento': lambda x: format_binario(x),
    'cancelamento/alteracao_destino': lambda x: format_binario(x),
    
    'intervalo_atraso': lambda x: format_intervalo(x, FAIXAS_ATRASO),
    'intervalo_extravio_temporario': lambda x: format_intervalo(x, FAIXAS_EXTRAVIO),
    
    # 'faixa_intervalo_atraso': lambda x: int(x),
    # 'faixa_intervalo_extravio_temporario': lambda x: int(x),

    # 'culpa_exclusiva_consumidor': lambda x: format_binario(x),
    # 'condicoes_climaticas/fechamento_aeroporto': lambda x: format_binario(x),
    # 'fechamento_aeroporto': lambda x: format_binario(x),
    
    # 'assistencia_cia_aerea': lambda x: format_binario(x, anomaly=-1),
    # 'dano_moral_individual': lambda x: format_comma_strings(x),
    # 'faixa_dano_moral_individual': lambda x: int(x),
}
