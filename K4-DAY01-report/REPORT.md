# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    "taxonomy_name": "ImageNet-1K",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
- Record này mô tả toàn ảnh như thế nào?
    + record này mô tả toàn bộ vật thể để phân loại và dự đoán lớp phù hợp cao nhất, và phân loại cao nhất là cab. Source là khoảng 0.510915 cho thấy là module có mức tự tin cao nhất
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    + Danh sách class được xác định bởi taxonomy của dataset để huấn luyện checkport và trong bài này thì checkport là ImageNet-1K
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    + vì class_id giúp hệ thống xữ lý và đối chiếu nhãn 1 cách nhất quan nhất, class_name giúp con người hiểu nhãn , còn taxonomy giúp xác định bộ nhãn mà ID và tên lớp thuộc về. Nhờ đó tránh nhầm lẫn khi các mô hình hoặc dataset sử dụng taxonomy khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    + Nếu ảnh có nhiều chủ thể, guideline cần quy định rõ tiêu chí chọn nhãn cấp ảnh, ví dụ xác định chủ thể chính theo mục tiêu của dataset. Annotator phải áp dụng cùng một quy tắc thay vì tự chọn theo cảm tính hoặc dựa trực tiếp vào prediction của model.
- Vì sao model score không phải ground truth?
    Model score chỉ thể hiện mức độ tự tin của model đối với prediction, không phải nhãn chuẩn. Ground truth phải được xác định theo annotation guideline và được kiểm tra độc lập với prediction của model.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    + class_name = person
    bbox_xyxy = [385.33, 69.24, 498.92, 348.92]
    bbox_width  = 113.58
    bbox_height = 279.68
- Diễn giải vị trí box bằng lời: 
    + Bounding box của person nằm ở khu vực bên phải của ảnh, từ khoảng x = 385 đến x = 499 và y = 69 đến y = 349. Box có chiều rộng khoảng 114 pixel và chiều cao khoảng 280 pixel, bao quanh phần lớn cơ thể của người được phát hiện.
- So sánh số prediction ở hai threshold:
    + Ở threshold 0.20 có 17 predictions, trong khi threshold 0.60 chỉ còn 6 predictions. Khi tăng threshold, các prediction có score thấp bị loại bỏ nên số lượng prediction giảm.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    + Threshold thấp làm tăng số prediction nên độ bao phủ cao hơn, nhưng reviewer phải kiểm tra nhiều prediction hơn. Threshold cao làm giảm khối lượng review nhưng có thể bỏ sót các object có confidence thấp.
- Đề xuất một quy tắc box chặt:
    + Bounding box nên ôm sát phần nhìn thấy của object, không lấy thừa nhiều background và không cắt mất phần object còn nhìn thấy.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    + Guideline cần quy định cách xử lý object bị che khuất hoặc cắt mép, ví dụ có gán nhãn hay không và box chỉ bao phần nhìn thấy hay theo quy tắc nào. Nếu trường hợp không được guideline mô tả rõ thì cần escalation cho reviewer hoặc người phụ trách quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - `instance_id`: `kitchen-001`
  - `class_name`: `person`
  - `score`: `0.899318`
  - Số điểm polygon: `348`
  - Một phần `polygon_xy`: `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], ...]`

- Polygon bổ sung chi tiết gì so với box?
  - Polygon mô tả đường biên và hình dạng của đối tượng chi tiết hơn bounding box. Bounding box chỉ là một hình chữ nhật bao quanh object, còn polygon có thể bám theo hình dạng thực tế của object và giúp phân biệt object với phần background bên trong box.

- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` dùng để phân biệt từng đối tượng riêng biệt trong ảnh, kể cả khi nhiều đối tượng cùng một class. Nó không phải `class_id` và cũng không phải ID dùng để tracking một object qua nhiều frame.

- Đề xuất một quy tắc biên mask:
  - Mask nên bám sát đường biên phần nhìn thấy của object, không lấy thừa background và không tự suy đoán phần object đang bị che khuất nếu guideline không yêu cầu.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định cách xác định biên khi object bị mờ, tiếp xúc với object khác hoặc bị che khuất, ví dụ có chỉ annotate phần nhìn thấy hay không và cách tách hai instance đang chạm nhau. Nếu không thể xác định rõ theo guideline thì cần escalation cho reviewer hoặc người phụ trách quyết định.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cấp ảnh | Ảnh có thể chứa nhiều chủ thể nên khó xác định nhãn chính | Chọn nhãn theo taxonomy và quy tắc trong guideline | Kiểm tra class có đúng và có tuân thủ guideline hay không |
| Phát hiện vật thể | Class và bounding box `[xmin, ymin, xmax, ymax]` cho từng object | Có thể bỏ sót object, chọn sai class hoặc box quá rộng/quá hẹp | Xác định đúng object, class và vẽ box ôm sát phần nhìn thấy | Kiểm tra class, vị trí box, object bị bỏ sót hoặc prediction thừa |
| Instance segmentation | Class, `instance_id` và polygon/mask cho từng object | Biên object có thể mờ, bị che khuất hoặc nhiều object tiếp xúc nhau | Tạo mask/polygon bám theo biên của từng instance theo guideline | Kiểm tra mask có đúng biên, có lẫn background và có tách đúng các instance hay không |


## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
