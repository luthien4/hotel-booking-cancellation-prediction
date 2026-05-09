<div style="background-color:  #f7dae3; padding: 10px;">

### Dataset Description

</div>

The data contains information about the demand for hotels. The dataset contains 119,390 observations. Each observation represents a hotel booking. The dataset includes bookings that were scheduled to arrive between July 1, 2015 and August 31, 2017, including both actual and cancelled bookings.

The data contains information about 68,163 hotels.

Training set - 44,638 rows.
Test set - 23,525 rows.

Files

train.csv - the training set
test.csv - the test set

| Column Name                   | Details                                                                                                                         |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `ADR`                         | Average price per room/night.                                                                                                   |
| `Adults`                      | Number of adults                                                                                                                |
| `Agent`                       | ID of the travel agency that made the booking                                                                                   |
| `ArrivalDateDayOfMonth`       |                                                                                                                                 |
| `ArrivalDateMonth`            |                                                                                                                                 |
| `ArrivalDateWeekNumber`       |                                                                                                                                 |
| `ArrivalDateYear`             |                                                                                                                                 |
| `AssignedRoomType`            | Code for the room type assigned to the reservation (1)                                                                          |
| `Babies`                      | Number of babies                                                                                                                |
| `BookingChanges`              | Number of changes/additions made to the reservation from the time it was entered into the system until check-in or cancellation |
| `Children`                    | Number of children                                                                                                              |
| `Company`                     | ID of the company/legal entity that made the reservation or is responsible for its payment                                      |
| `Country`                     | Country                                                                                                                         |
| `CustomerType`                | Reservation type, which can be one of four categories: Contract, Group, Transient, Transient-party                              |
| `DaysInWaitingList`           | Number of days the reservation was on the waiting list before it was confirmed to the customer                                  |
| `DepositType`                 | Indication of whether the customer has paid a deposit to guarantee the reservation (2)                                          |
| `DistributionChannel`         | Reservation distribution channel. The term "TA" stands for "Travel Agents" and "TO" stands for "Tour Operators"                 |
| `IsCanceled`                  | Value indicating whether the booking was cancelled (1) or not (0)                                                               |
| `IsRepeatedGuest`             | Value indicating whether the booking name was from a repeat guest (1) or not (0)                                                |
| `LeadTime`                    | Number of days between the date the booking was entered into the system and the arrival date                                    |
| `MarketSegment`               | Market segment designation. In categories, the term "TA" stands for "Travel Agents" and "TO" stands for "Tour Operators"        |
| `Meal`                        | The type of meal booked (3)                                                                                                     |
| `PreviousBookingsNotCanceled` | Reservation type, which can be one of four categories: Contract, Group, Transient, Transient-party                              |
| `PreviousCancellations`       | Number of previous bookings that were cancelled by the customer prior to the current booking                                    |
| `RequiredCardParkingSpaces`   | Number of parking spaces required by the customer                                                                               |
| `ReservationStatusDate`       | Date when the last status was set (4)                                                                                           |
| `ReservedRoomType`            | The room type code reserved                                                                                                     |
| `StaysInWeekendNights `       | Number of weekend nights (Saturday or Sunday) that the guest stayed or booked to stay at the hotel                              |
| `StaysInWeekNights`           | Number of weeknights (Monday through Friday) that the guest stayed or booked to stay at the hotel                               |
| `TotalOfSpecialRequests`      | Number of special requests made by the customer (e.g. twin beds or a higher floor)                                              |


(1) Sometimes the assigned room type differs from the reserved room type due to hotel operations (e.g. overbooking) or at the customer's request

(2) This variable can take three categories:
        No Deposit - no deposit has been paid
        Non Refund - a deposit equal to the full cost of the stay has been paid
        Refundable - a deposit equal to less than the total cost of the stay has been paid

(3) The categories are represented in standard hospitality packages:
        Undefined / SC - no meals;
        BB - breakfast;
        HB - breakfast and one other meal - usually dinner);
        FB - breakfast, lunch and dinner

(4) This variable can be used together with ReservationStatus to understand when the reservation was cancelled or when the customer checked out of the hotel.