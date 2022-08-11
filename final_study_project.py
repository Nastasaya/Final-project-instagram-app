#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import pandas as pd
import scipy.stats as ss
import seaborn as sns


# Проект: вариант 4
# 
#  Задание 1
# Представьте, что вы аналитик в компании, которая разрабатывает приложение для обработки и оформления фотографий в формате Stories (например, для дальнейшего экспорта в Instagram Stories). Был проведен A/B тест: тестовой группе предлагалась новая модель оплаты коллекций шаблонов, контрольной – старая механика. Ваша основная задача: проанализировать итоги эксперимента и решить, нужно ли выкатывать новую модель на остальных пользователей.
# 
# В ходе отчета обоснуйте выбор метрик, на которые вы обращаете внимание. Если различия есть, то объясните, с чем они могут быть связаны и являются ли значимыми.
# 
# Данные:
# 
# - active_users – информация о пользователях, которые посещали приложение во время эксперимента. 
# - groups – разбивка пользователей на контрольную (А) и тестовую (В) группы. 
# - purchases – данные о транзакциях (оплатах) пользователей приложения во время эксперимента 

# In[2]:


users = pd.read_csv('Проект_4_active_users.csv', sep=',')
users.head()


# In[3]:


users.shape


# In[4]:


users.nunique()


# In[5]:


groups_AB = pd.read_csv('Проект_4_groups.csv', sep=',')
groups_AB.head()


# In[6]:


groups_AB.shape


# In[7]:


purchases = pd.read_csv('Проект_4_purchases.csv', sep=',')
purchases.head()


# In[8]:


purchases.shape


# In[9]:


df = users.merge(groups_AB, on='user_id', how='left').merge(purchases, how='left', on='user_id')
df.head()


# In[10]:


# Заменим NaN на 0
df.revenue.fillna(0, inplace=True)


# In[11]:


df.groupby('group', as_index=False).agg({'user_id': 'count'})


# Проведем небольшой эксплораторный анализ. Для этого посмотрим тип данных датафрейма, его размер, проверим отсутствующие значения и проверим основные статистические показатели.

# In[12]:


df.shape


# Датафрейм включает в себя 8341 строку и 6 столбцов (при этом количество пользователей совершивших оплату всего 541). Для проверки гипотезы об успешности новой модели оплаты, мы будем учитывать только активных пользователей (пользователей попавших под тестирование было 74576, но мы будем учитывать только активных пользователей.

# In[13]:


df.dtypes


# In[14]:


df.isna().sum()
# в датафрейме отсутствуют пропущенные значения


# In[15]:


df.describe().round(2)


# Рассчитаем conversion rate (CR) для каждой из групп и посмотрим на уровень конверсии в покупку для каждой. Но перед этим рассчитаем число активных пользователей и пользователей, которые произвели оплату в разрезе групп:

# In[16]:


users_A = df.query('group == "A"')['user_id'].count()
users_B = df.query('group == "B"')['user_id'].count()
users_A_pay = df.query("group == 'A' and revenue > 0")['user_id'].count()
users_B_pay = df.query("group == 'B' and revenue > 0")['user_id'].count()
print('group A', users_A)
print('group B', users_B)
print('group A pay', users_A_pay)
print('group B pay', users_B_pay)


# In[17]:


CR_A = (users_A_pay / users_A * 100).round(2)
CR_B = (users_B_pay / users_B * 100).round(2)
print('CR group A', CR_A, 'CR group B', CR_B)


# По итогам расчета видно, что уровень конверсии в тестовой группе ниже, чем в контрольной. Для того, чтобы понять так ли это, мы проведем статистическую оценку вероятности.
# Поскольку конверсия имеет дискретное распределение вероятностей, вероятность которого равна либо 1, либо 0. Пользователь либо оплачивает покупку (1), либо не оплачивает (0).
# Раз мы хотим проверить гипотезу по конверсии, которая является категориальной переменной, соответствующей дискретному распределению, мы рассчитаем chi2 и проверим нулевую гипотезу: распределение пользователей, которые платят и не платят в контрольной и тестовой группе не различается.

# In[18]:


df['pay_notpay'] = df.revenue.apply(lambda x: 'pay' if x > 0 else 'notpay')


# In[19]:


pd.crosstab(df.pay_notpay, df.group, values=df.user_id, aggfunc='count')


# In[20]:


stat, p, dof, expected = chi2_contingency(pd.crosstab(df.pay_notpay, df.group, values=df.user_id, aggfunc='count'))
stat, p


# In[21]:


"""prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Отклоняем H0')
else:
    print('Не отклоняем H0')"""


# In[22]:


prob = 0.95
alpha = 1.0 - prob
if p <= alpha:
    print('Отклоняем H0')
else:
    print('Не отклоняем H0')


# Предположим, что основным фактором для принятия решения о применении новой модели на всех пользователях, будет являтся важный показатель выручки. Именно поэтому мы будет смотреть среднее значение revenue и проверять гипотезу групп А и В в разрезе выручки. 
# Для начала рассчитаем среднее значение показателя revenue для контрольной и тестовой групп:

# In[23]:


pd.crosstab(df.user_id, df.group, values=df.revenue,
            aggfunc='sum').mean()


# Далее стоит посмотреть на распределение выручки по группам.

# In[24]:


df_rev = df.query('revenue > 0')
# исключим пользователей, которые не платили


# In[25]:


sns.distplot(df_rev.revenue)


# Видно, что распредление несколько отклоняется от нормального и возможно присутствуют выбросы. Поэтому прежде чем проверять нулевую гипотезу, используя параметрические или непараметрические критерии оценки, нужно проверить нормальность распределения выборки, для этого построим гистограмму и рассчитаем критерий Шапиро-Уилка:

# In[26]:


ss.shapiro(df_rev.query('group == "A"').revenue)
# p-value < 0.05, следовательно распределение не является нормальным


# In[27]:


ss.shapiro(df_rev.query('group == "B"').revenue)
# p-value < 0.05, следовательно распределение не является нормальным


# Несмотря на результаты расчета критерия Шапиро-Уилка, учитывая большое количество наблюдений в выборке, результаты расчета критерия Стьюдента считаем действительными (https://stats.stackexchange.com/questions/9573/t-test-for-non-normal-when-n50)

# In[28]:


ttest_ind(df_rev.query('group == "A"').revenue,
          df_rev.query('group == "B"').revenue)


# Ввиду того, что p-value < 0.05, мы отклоняем нулевую гипотезу. К тому же среднее значение оплат в тестовой группе выше среднего значения в контрольной группе, что тоже говорит о возможной успешности новой модели.
# Ввиду всего вышеуказанного можно было бы сказать, что эксперимент успешен и можно выкатывать новую модель на остальных пользователей. Но для того, чтобы не принять необдуманных решений, стоит рассмотреть выборку в разрезе имеющихся показателей и попробовать применить еще один непараметрический метод, поскольку распределение в выборке не является нормальным.
# Используем непараметрический метод – U-критерий Манна-Уитни. Он проранжирует все данные revenue и рассчитает какой средний ранг оказался в первой группе и какой во второй. Этот критерий менее чувствителен к экстремальным отклонениям от нормальности и наличию выбросов.

# In[29]:


RA = df_rev.query('country == "Russia" and group == "A"').revenue
RB = df_rev.query('country == "Russia" and group == "B"').revenue
SA = df_rev.query('country == "Sweden" and group == "A"').revenue 
SB = df_rev.query('country == "Sweden" and group == "B"').revenue
AA = df_rev.query('platform == "android" and group == "A"').revenue
AB = df_rev.query('platform == "android" and group == "B"').revenue
IA = df_rev.query('platform == "ios" and group == "A"').revenue
IB = df.query('platform == "ios" and group == "B"').revenue
FA = df_rev.query('sex == "female" and group == "A"').revenue
FB = df_rev.query('sex == "female" and group == "B"').revenue
MA = df_rev.query('sex == "male" and group == "A"').revenue
MB = df_rev.query('sex == "male" and group == "B"').revenue
print(mannwhitneyu(RA, RB))
print(mannwhitneyu(SA, SB))
print(mannwhitneyu(AA, AB))
print(mannwhitneyu(IA, IB))
print(mannwhitneyu(FA, FB))
print(mannwhitneyu(MA, MB))


# По итогам этого анализа можно увидеть более интересные результаты. Так, например, при группировке по половому признаку - очевидно, что нулевую гипотезу для мужчин мы отклонить не можем, соответственно и выкатывать на эту группу новую модель не имеет смысла.
# 
# Но стоит помнить о выбросах в выборке, поэтому лучшим решением будет запросить новую выборку, проверить ее на наличие выбросов и уже тогда принимать решение.

# Ссылка на дашборд
# https://public.tableau.com/app/profile/anastasia1983/viz/final_project4_a-poplavskaja-20_corr/Users_rev
