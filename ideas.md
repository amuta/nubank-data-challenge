
** REMEMBER TO DO TRANSFORMATIONS INSIDE CROSSVALIDATION ** 

=data

-encoding ? 1-1mapping? target encoding?
--I only labeled categories with more than 10 entries and put the rest into a "other" group, which increased the metric.

--https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

--https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
. I think (and this is just a personal opinion) for categorical variables with many classes, one-hot encoding is the safest approach because it does not impose arbitrary values to categories. The only downside to one-hot encoding is that the number of features (dimensions of the data) can explode with categorical variables with many categories. To deal with this, we can perform one-hot encoding followed by PCA or other dimensionality reduction methods to reduce the number of dimensions (while still trying to preserve information).


- ALWAYS DO encoding in data before CV??
?
=features


- reported income seems really bad

- compare ID vs test ? there shoudn't be no same ids?

- https://www.kaggle.com/maximilianhahn/manager-skill-for-cross-validation-pipelines

- tipo de rua consulta ZIP

- distancia ZIP e lat_lon

- COUNT NA
- NA BY COLUMN (some NAs are row-wise correlated)

Coluna	Tipo	Descrição
ids	String	identificador único de um aplicante
email	String	Provedor de e-mail do solicitante
tags	String	Tags descritivas dadas pelo provedor de dados
score_1	String	Score de crédito 1, categorias
score_2	String	Score de crédito 2, categorias
score_3	Float	Score de crédito 3
score_4	Float	Score de crédito 4
score_5	Float	Score de crédito 5
score_6	Float	Score de crédito 6
risk_rate	Float	Risco associado ao aplicante
last_amount_borrowed	Float	Valor do último empréstimo que o aplicante tomou
last_borrowed_in_months	Int	Duração do último empréstimo que o aplicante tomou
reason	String	Razão pela qual foi feita uma consulta naquele cpf
income	Float	Renda estimada pelo provedor dos dados para o aplicante
facebook_profile	Bool	Se o aplicante possui perfil no Facebook
state	String	Estado de residência do aplicante
zip	String	Código postal do aplicante
shipping_zip_code	Int	Código do endereço de entrega
shipping_state	String	Estado do endereço de entrega
channel	String	Canal pelo qual o aplicante aplicou
job_name	String	Profissão do aplicante
real_state	String	Informação sobre habitação do aplicante
ok_since	Float	Quantidade de dias que
n_bankruptcies	Float	Quantidade de bancarrotas que o aplicante já experimentou
n_defaulted_loads	Float	Quantidade de empréstimos não pagos no passado
n_accounts	Float	Número de contas que o aplicante possui
n_issues	Float	Número de reclamações de terceiros feitas em alguma das contas do aplicante
user_agent	String	Informação sobre dispositivo usado para a aplicação
reported_income	Int	Renda informada pelo próprio aplicante
profile_phone_number	String	Número de telefone, ex: 210-2813414
marketing_channel	String	Canal de marketing pelo qual o aplicante chegou na página de pedido de crédito
lat_lon	Object	Latitude e longitude da localização
external_data_provider_fraud_score	Int	Score de fraude
external_data_provider_first_name	String	Primeiro nome do aplicante
external_data_provider_email_seen_before	String	Se o e-mail já foi consultado junto ao provedor de dados
external_data_provider_credit_checks_last_year	Int	Quantidade de consultas de crédito na janela de um ano
external_data_provider_credit_checks_last_month	Int	Quantidade de consultas de crédito na janela de um mês
external_data_provider_credit_checks_last_2_year	Int	Quantidade de consultas de crédito na janela de dois anos
application_time_in_funnel	Int	Tempo gasto pelo aplicante durante o processo de aplicação
application_time_applied	Date	Horário de aplicação
target_default	Bool	Indicativo de default
target_fraud	String	Pode assumir dois valores positivos referentes a dois tipos de fraude: fraud_id/fraud_friends_family, NaN se não houve


Dados de comportamento
Coluna  Tipo    Descrição
ids String  identificador único de um aplicante
credit_line Int Limite do cartão
month   Int Ordenação dos meses que a pessoa é cliente, sendo 0 o primeiro mês dela como cliente
spend   Float   Valor gasto naquele mês
revolving_balance   Float   Valor que o cliente não pagou da fatura atual e que irá rolar para a próxima
card_request    Int Se o cliente solicitou uma nova via do cartão (ou a primeira)
minutes_cs  Float   Quantidade de minutos utilizados do serviço de atendimento ao consumidor