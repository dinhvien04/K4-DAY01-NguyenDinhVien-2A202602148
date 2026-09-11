# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.5 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  - `taxonomy_name`: "ImageNet-1K"
  - `rank`: 1
  - `class_id`: 468
  - `class_name`: "cab"
  - `score`: 0.510915
- Record này mô tả toàn ảnh như thế nào?
  - Record này gán một nhãn phân loại cho toàn bộ bức ảnh (image-level prediction), không khoanh vùng hay xác định tọa độ của từng vật thể riêng lẻ. Model dự đoán nhãn có độ tự tin cao nhất cho toàn ảnh là `cab` với score khoảng 0.510915.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  - Danh sách class được xác định bởi taxonomy của dataset dùng để huấn luyện checkpoint, cụ thể trong bài này là taxonomy ImageNet-1K (1.000 lớp).
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - Vì `class_id` giúp hệ thống máy tính xử lý và đối chiếu dữ liệu một cách nhất quán, `class_name` giúp con người đọc hiểu nhãn trực quan, còn `taxonomy_name` giúp xác định bộ nhãn ngữ cảnh mà ID và class name thuộc về, tránh nhầm lẫn khi các mô hình hoặc dataset khác nhau sử dụng cùng mã số nhưng khác ý nghĩa.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  - Nếu ảnh có nhiều chủ thể, guideline cần quy định rõ tiêu chí chọn nhãn cấp ảnh (ví dụ: ưu tiên chủ thể chiếm diện tích lớn nhất, chủ thể nằm ở trung tâm, hoặc chủ thể phục vụ mục tiêu chính của bài toán). Annotator phải áp dụng cùng một quy tắc nhất quán thay vì tự chọn theo cảm tính hoặc phụ thuộc vào prediction của model.
- Vì sao model score không phải ground truth?
  - Model score chỉ thể hiện mức độ tự tin (confidence) của mô hình đối với dự đoán của chính nó, không phải là nhãn chuẩn xác thực tế. Ground truth phải do con người xác định độc lập dựa trên annotation guideline và quy trình kiểm thử/QC.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  - `class_name`: `person`
  - `score`: `0.912625`
  - `bbox_xyxy`: `[385.33, 69.24, 498.92, 348.92]`
  - `bbox_width`: `113.58`
  - `bbox_height`: `279.68`
- Diễn giải vị trí box bằng lời:
  - Bounding box của `person` nằm ở khu vực bên phải của ảnh, từ khoảng x = 385.33 đến x = 498.92 và y = 69.24 đến y = 348.92 (gốc tọa độ 0,0 ở góc trên bên trái). Box có chiều rộng khoảng 113.58 pixel và chiều cao khoảng 279.68 pixel, bao quanh phần lớn cơ thể nhìn thấy của người trong ảnh.
- So sánh số prediction ở hai threshold:
  - Ở threshold 0.20 có 17 predictions, trong khi ở threshold 0.60 chỉ còn 6 predictions. Khi tăng threshold, các prediction có score thấp bị loại bỏ nên số lượng prediction giảm đi rõ rệt.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Threshold thấp (0.20) giúp tăng độ bao phủ (coverage), giảm nguy cơ bỏ sót đối tượng nhưng reviewer phải tốn nhiều công sức để lọc và loại bỏ các prediction rác (false positives). Ngược lại, threshold cao (0.60) giảm tải khối lượng review nhưng có thể bỏ sót các đối tượng thực tế có độ tự tin thấp (false negatives).
- Đề xuất một quy tắc box chặt:
  - Bounding box phải ôm sát đường biên nhìn thấy của đối tượng, tiếp xúc với các điểm cực biên (trên, dưới, trái, phải), không bao gồm diện tích background thừa và không được cắt lẹm vào các phần cơ thể/vật thể nhìn thấy được.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định rõ tỷ lệ nhìn thấy tối thiểu để gán nhãn (ví dụ: thấy trên 20% mới gán nhãn), quy tắc vẽ box chỉ bao phần nhìn thấy hay ước lượng toàn bộ vật thể. Nếu gặp trường hợp vật thể bị che khuất quá nhiều hoặc cắt mép không thể nhận diện chắc chắn theo guideline thì annotator phải escalate cho reviewer hoặc lead quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - `instance_id`: `kitchen-001`
  - `class_name`: `person`
  - `score`: `0.899318`
  - Số điểm polygon: `348`
  - Một phần `polygon_xy`: `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], ...]`

- Polygon bổ sung chi tiết gì so với box?
  - Polygon mô tả chính xác hình dạng hình học và đường biên thực tế của đối tượng theo từng pixel. Bounding box chỉ là một hình chữ nhật bao ngoài (chứa cả các pixel background ở các góc), còn polygon bám sát đường cong viền ngoài của đối tượng, giúp phân biệt rõ từng pixel thuộc về đối tượng hay background.

- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` dùng để định danh và phân biệt từng cá thể (instance) riêng biệt trong cùng một ảnh, kể cả khi chúng thuộc cùng một class. Nó không phải là `class_id` (mã lớp chung) và cũng không phải là tracking ID (định danh theo dõi xuyên suốt qua nhiều khung hình/video).

- Đề xuất một quy tắc biên mask:
  - Đường biên mask/polygon phải bám sát theo rìa nhìn thấy của đối tượng với độ sai lệch không quá 2-3 pixel, không bao gồm background và không tự suy đoán phần bị che khuất trừ khi có quy định riêng trong guideline.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần hướng dẫn rõ cách phân chia đường ranh giới khi hai instance cùng lớp chạm nhau hoặc chồng lấn, và xử lý vùng biên bị mờ do chuyển động hoặc ánh sáng. Trường hợp không phân định được ranh giới rõ ràng phải escalate lên reviewer để thống nhất quy tắc xử lý chung cho toàn đội.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cấp ảnh (`class_id`, `class_name`) kèm taxonomy | Ảnh chứa nhiều chủ thể cùng lúc (ví dụ ảnh giao thông có cả xe buýt, ô tô con, người đi bộ) | Chọn nhãn theo đúng tiêu chí ưu tiên được mô tả trong guideline | Kiểm tra nhãn có thuộc đúng taxonomy và tuân thủ đúng quy tắc chọn chủ thể của guideline hay không |
| Phát hiện vật thể | Class và bounding box `[xmin, ymin, xmax, ymax]` (pixel) cho từng vật thể | Bỏ sót vật thể nhỏ ở hậu cảnh, chọn nhầm class hoặc box vẽ quá rộng/lẹm vào vật thể | Xác định đúng vị trí từng vật thể, chọn đúng class và vẽ box ôm sát phần nhìn thấy | Soát các box thừa, phát hiện vật thể bị bỏ sót, kiểm tra độ chặt của box và tính chính xác của class |
| Instance segmentation | Class, `instance_id` và polygon/mask pixel cho từng cá thể | Đường biên khó xác định do bị che khuất, mờ biên hoặc hai vật thể dính liền nhau | Vẽ polygon tỉ mỉ men theo đường biên thực của từng instance theo guideline | Kiểm tra độ khít của biên mask, kiểm tra xem có pixel background bị gộp nhầm và tách đúng các instance riêng biệt hay không |


## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không tải lên Colab hoặc đưa vào bài nộp công khai các dữ liệu cá nhân, khuôn mặt, biển số xe, ảnh riêng tư hoặc dữ liệu nội bộ/khách hàng; chỉ sử dụng các tập dữ liệu mẫu công khai có giấy phép phù hợp (CC BY 2.0).
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Giảng viên hướng dẫn / Lab Coach / Người phụ trách quản trị dữ liệu.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
