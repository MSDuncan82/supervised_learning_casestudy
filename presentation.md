# Supervised Learning Case Study: <br> Predicting Rider Churn
Mike, Jacob, Clayton, Abel

## Introduction & Goal

* 50,000 riders who signed up in January of 2014
* 10,000 riders were randomly allocated unseen dataset: churn_test.csv

## Data Description & Impution 

* Seems to be fictional dataset, or the city names were purposely obscured. City names included: Winterfell, King's Landing and Astapor.
* Column Names: avg_dist, avg_rating_by_driver, avg_rating_of_driver, avg_surge, city, last_trip_date, phone, signup_date, surge_pct, trips_in_first_30_days, luxury_car_user, weekday_pct

* **Churn was assumed to be riders that have not taken a ride in the last 30 days since Date of Reference: June 1, 2014**

* Blank Data (Overall, Train, Test):
    * avg_rating_by_driver: 201, 162, 39
    * avg_rating_of_driver: 8122, 6528, 1594
    * phone:                396, 319, 77

* In churn_train, to impute:
    * **ratings:**  null = mean
    * **phones:**   null = 1 (aka iPhone)
        * There were 27947 iPhones, 12053 Andriods in churn_train, therefore we imputed the most popular

## Goal

* We would like to use this dataset to help understand what factors are the best **predictors for retention**, and offer suggestions to help Company X. 

* Build a **model that minimizes error**, but also a model that allows you to **interpret the factors** that contributed to your predictions.

## Exploritory Data Analysis


## Modeling 


## Conclusions

