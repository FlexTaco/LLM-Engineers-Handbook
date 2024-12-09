# RAG Final Project

### Environment and Tooling

[docker-compose.yml](https://github.com/FlexTaco/LLM-Engineers-Handbook/blob/main/docker-compose.yml)

I don't know how to add screenshots here but they are in the assets folder.
![image](https://github.com/user-attachments/assets/Screenshot 2024-12-09 at 4.36.54 AM.png)

### ETL Pipeline

URLs used

- https://github.com/ros2/ros2_documentation
- https://github.com/gazebosim/docs.git
- https://github.com/ros-navigation/docs.nav2.org.git
- https://github.com/moveit/moveit2_tutorials.git

Raw data are stored in `mongodb` using `poetry poe run-digital-data-etl`.

### Featurization Pipeline

Raw data can be cleaned, chunked, embedded into the vector db using `poetry poe run-feature-engineering-pipeline`.
